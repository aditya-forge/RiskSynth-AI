# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Client profile generator trained on the INE household table.

Pipeline (configs/model_config.yaml -> synthetic_generation.pipeline):
  stage 1  a model learns the INE household base (wealth_profile.build_stage1_table):
           bootstrap sampler, or SDV CTGAN / Gaussian copula / TVAE when SDV is installed
  stage 2  wealth features + CR01-CR07 / MiFID II enforcement (risksynth.engine.stage2)
  stage 3  edge-case cohort EC01-EC09 (risksynth.engine.edge_cases)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Literal
from datetime import datetime
import yaml
import json

try:
    from loguru import logger
    _HAS_LOGURU = True
except ImportError:  # loguru is optional; fall back to stdlib logging
    import logging
    logger = logging.getLogger("client_generator")
    _HAS_LOGURU = False

from risksynth.engine import wealth_profile
from risksynth.engine.stage2 import run_stage2
from risksynth.paths import LOGS_DIR, MODEL_CONFIG, SYNTHETIC_DIR

LEGACY_REQUIRED = ['edad', 'nivel_estudios', 'situacion_laboral', 'ingresos_anuales', 'saldo_cuenta']
WEALTH_REQUIRED = ['edad', 'situacion_laboral', 'net_income_annual_eur',
                   'household_expenses_annual_eur', 'debt_service_annual_eur']


def _sdv():
    """Import SDV lazily so the module (and stage 2) work without it installed."""
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    return CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer, SingleTableMetadata


def sdv_available() -> bool:
    try:
        _sdv()
        return True
    except ImportError:
        return False


class BootstrapSynthesizer:
    """Baseline stage-1 'model': weighted resampling of observed INE households.

    No generative model and no new household combinations; stage 2 still derives fresh
    wealth features. Useful as a fidelity reference and when SDV is not installed.
    Same fit/sample/save/load interface as the SDV synthesizers.
    """

    def __init__(self, weight_column: str = "peso", seed: int = 42):
        self.weight_column = weight_column
        self.seed = seed
        self.table: Optional[pd.DataFrame] = None
        self._draws = 0

    def fit(self, df: pd.DataFrame) -> None:
        self.table = df.reset_index(drop=True)

    def sample(self, num_rows: int) -> pd.DataFrame:
        if self.table is None:
            raise ValueError("fit() first")
        weights = self.table[self.weight_column] if self.weight_column in self.table else None
        self._draws += 1
        out = self.table.sample(n=num_rows, replace=True, weights=weights,
                                random_state=self.seed + self._draws)
        return out.reset_index(drop=True)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BootstrapSynthesizer":
        import pickle
        with open(path, "rb") as f:
            model: BootstrapSynthesizer = pickle.load(f)
        return model


class ClientGenerator:
    """
    Generador de datos sintéticos de clientes bancarios.
    
    Attributes:
        n_samples: Número de clientes sintéticos a generar
        method: Método de generación ('ctgan', 'copula', 'vae')
        config_path: Ruta al archivo de configuración
        seed: Semilla para reproducibilidad
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        method: Literal['ctgan', 'copula', 'vae', 'bootstrap'] = 'ctgan',
        config_path: Optional[str] = None,
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.method = method
        self.seed = seed
        self.config_path = config_path or str(MODEL_CONFIG)
        self.config = self._load_config(config_path)
        self.synthesizer = None
        self.metadata = None
        self.wealth_report = None
        
        # Configurar logger (fichero solo con loguru)
        if _HAS_LOGURU:
            (LOGS_DIR / "model_training").mkdir(parents=True, exist_ok=True)
            logger.add(
                str(LOGS_DIR / "model_training" / f"client_generator_{datetime.now():%Y%m%d_%H%M%S}.log"),
                rotation="500 MB"
            )
        
        logger.info(f"Inicializado ClientGenerator con metodo '{method}'")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Cargar configuración desde archivo YAML."""
        if config_path is None:
            config_path = str(MODEL_CONFIG)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config['synthetic_generation']['methods'][self.method]
    
    def load_real_data(self, data_path: str) -> pd.DataFrame:
        """
        Cargar datos reales para entrenar el generador.
        
        Args:
            data_path: Ruta al archivo CSV con datos reales
            
        Returns:
            DataFrame con datos reales procesados
        """
        logger.info(f"Cargando datos reales desde: {data_path}")
        
        path = Path(data_path)
        if path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        logger.info(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        
        # Validación básica
        self._validate_data(df)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validar que los datos cumplen requisitos mínimos.

        Acepta el esquema de crédito (1.x) o la tabla de hogares INE de RiskSynth (2.0).
        """
        if set(WEALTH_REQUIRED) <= set(df.columns):
            required_columns = list(WEALTH_REQUIRED)
        else:
            required_columns = list(LEGACY_REQUIRED)
            # 'provincia' o 'ccaa' son válidos como columna geográfica
            if 'provincia' not in df.columns and 'ccaa' not in df.columns:
                required_columns.append('provincia')

        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing} "
                             f"(esquema wealth: {WEALTH_REQUIRED})")
        
        # Verificar valores nulos
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Columnas con valores nulos:\n{null_counts[null_counts > 0]}")
        
        logger.info("[OK] Validacion de datos completada")
    
    def prepare_metadata(self, df: pd.DataFrame):
        """Preparar metadatos para SDV."""
        *_, SingleTableMetadata = _sdv()
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # Ajustes manuales (solo columnas presentes)
        sdtypes = {'edad': 'numerical', 'provincia': 'categorical', 'ccaa': 'categorical',
                   'nivel_estudios': 'categorical', 'situacion_laboral': 'categorical',
                   'contract_type': 'categorical', 'ingresos_anuales': 'numerical',
                   'saldo_cuenta': 'numerical'}
        for column, sdtype in sdtypes.items():
            if column in df.columns:
                metadata.update_column(column, sdtype=sdtype)

        self.metadata = metadata
        logger.info("[OK] Metadatos preparados")
        
        return metadata
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Entrenar el modelo generativo.
        
        Args:
            df: DataFrame con datos reales
        """
        logger.info(f"Entrenando modelo {self.method.upper()}...")

        if self.method == 'bootstrap':   # baseline sin SDV
            self.synthesizer = BootstrapSynthesizer(seed=self.seed)
            self.synthesizer.fit(df)
            logger.info(f"[OK] Bootstrap listo ({len(df)} hogares)")
            return

        # Preparar metadatos
        if self.metadata is None:
            self.prepare_metadata(df)
        CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer, _ = _sdv()

        # Inicializar synthesizer según método
        if self.method == 'ctgan':
            self.synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                epochs=self.config['hyperparameters']['epochs'],
                verbose=self.config['hyperparameters']['verbose']
            )
        
        elif self.method == 'copula':
            self.synthesizer = GaussianCopulaSynthesizer(
                metadata=self.metadata,
                default_distribution=self.config['hyperparameters']['default_distribution']
            )
        
        elif self.method == 'vae':
            self.synthesizer = TVAESynthesizer(
                metadata=self.metadata,
                epochs=self.config.get('hyperparameters', {}).get('epochs', 100),
                verbose=self.config.get('hyperparameters', {}).get('verbose', True)
            )
        else:
            raise ValueError(f"Método no soportado: {self.method}. Opciones: 'ctgan', 'copula', 'vae'")
        
        # Entrenar
        start_time = datetime.now()
        self.synthesizer.fit(df)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"[OK] Entrenamiento completado en {training_time:.2f} segundos")
    
    def generate(
        self,
        apply_constraints: bool = True,
        apply_post_processing: bool = True,
        apply_wealth_stage: bool = True,
        wealth_config: Optional[Dict] = None,
        edge_case_share: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generar datos sintéticos.

        Args:
            apply_constraints: Si aplicar restricciones de negocio
            apply_post_processing: Si aplicar post-procesamiento
            apply_wealth_stage: Stages 2+3 de RiskSynth (features wealth + CR01-CR07 + MiFID II
                + cohorte edge cases EC01-EC09, según edge_case_profiles.enabled).
                Se omite si las muestras no tienen las columnas de hogar (esquema 1.x).
            wealth_config: data_config ya cargado (p. ej. otro risk mix); None = configs/.
            edge_case_share: override del 8% total de edge cases.

        Returns:
            DataFrame con datos sintéticos
        """
        if self.synthesizer is None:
            raise ValueError("Debe entrenar el modelo primero con fit()")
        
        logger.info(f"Generando {self.n_samples} clientes sinteticos...")
        
        # Generar
        synthetic_df = self.synthesizer.sample(num_rows=self.n_samples)
        
        # Aplicar restricciones
        if apply_constraints:
            synthetic_df = self._apply_constraints(synthetic_df)
        
        # Post-procesamiento
        if apply_post_processing:
            synthetic_df = self._post_process(synthetic_df)

        # Stage 2: features wealth/robo-advisory + reglas de consistencia
        if apply_wealth_stage:
            if set(WEALTH_REQUIRED) <= set(synthetic_df.columns):
                synthetic_df, self.wealth_report = run_stage2(
                    synthetic_df, seed=self.seed + self._stage_seed_offset(),
                    cfg=wealth_config, edge_case_share=edge_case_share)
                logger.info(f"Stage 2 wealth: violations raw {self.wealth_report['violations_raw']} "
                            f"-> enforced {self.wealth_report['violations_enforced']}, "
                            f"exempted {self.wealth_report['violations_exempted']}")
                edge = self.wealth_report.get("edge_cases")
                if edge:
                    logger.info(f"Stage 3 edge cases: {edge['achieved_total_share_pct']:.2f}% "
                                f"(target {edge['target_total_share_pct']:.2f}%), "
                                + ", ".join(f"{k}={v['rows']}" for k, v in edge['profiles'].items()))
            else:
                logger.warning(f"Stage 2 omitido: faltan columnas {set(WEALTH_REQUIRED) - set(synthetic_df.columns)}")

        logger.info(f"[OK] Generacion completada: {len(synthetic_df)} registros")

        return synthetic_df

    def _stage_seed_offset(self) -> int:
        """seed_offset de stage_2_wealth_features en model_config.yaml (1000 por defecto)."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                pipeline = yaml.safe_load(f)['synthetic_generation']['pipeline']
            return int(pipeline['stage_2_wealth_features']['seed_offset'])
        except (OSError, KeyError, TypeError):
            return 1000
    
    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplicar restricciones de negocio."""
        logger.info("Aplicando restricciones de negocio...")
        
        # Edad mínima
        df['edad'] = df['edad'].clip(lower=18)

        # Importes no negativos (esquema 1.x y tabla de hogares 2.0)
        for column in ['saldo_cuenta', 'ingresos_anuales', 'net_income_annual_eur',
                       'household_expenses_annual_eur', 'debt_service_annual_eur',
                       'main_residence_value_eur', 'n_miembros', 'dependent_children']:
            if column in df.columns:
                df[column] = df[column].clip(lower=0)

        # Ratio deuda/ingreso
        if 'deuda_total' in df.columns and 'ingresos_anuales' in df.columns:
            max_deuda = df['ingresos_anuales'] * 0.4
            df['deuda_total'] = np.minimum(df['deuda_total'], max_deuda)
        
        logger.info("[OK] Restricciones aplicadas")
        return df
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-procesamiento de datos generados."""
        logger.info("Aplicando post-procesamiento...")
        
        # Redondeos
        df['edad'] = df['edad'].round().astype(int)
        if 'ingresos_anuales' in df.columns:
            df['ingresos_anuales'] = (df['ingresos_anuales'] / 100).round() * 100
        if 'saldo_cuenta' in df.columns:
            df['saldo_cuenta'] = df['saldo_cuenta'].round(2)
        for column in ['n_miembros', 'dependent_children']:
            if column in df.columns:
                df[column] = df[column].round()

        # Categorías válidas
        if 'provincia' in df.columns:
            valid_provinces = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza"]
            df['provincia'] = df['provincia'].apply(
                lambda x: x if x in valid_provinces else np.random.choice(valid_provinces)
            )
        
        logger.info("[OK] Post-procesamiento completado")
        return df
    
    def save(
        self,
        output_dir: str,
        synthetic_df: pd.DataFrame,
        save_model: bool = True
    ) -> Dict[str, str]:
        """
        Guardar datos sintéticos y modelo.
        
        Args:
            output_dir: Directorio de salida
            synthetic_df: DataFrame con datos sintéticos
            save_model: Si guardar el modelo entrenado
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar datos
        data_path = output_path / "clients_synthetic.csv"
        synthetic_df.to_csv(data_path, index=False)
        logger.info(f"[OK] Datos guardados en: {data_path}")
        
        # Guardar modelo
        model_path = None
        if save_model and self.synthesizer is not None:
            model_path = output_path / f"model_{self.method}.pkl"
            self.synthesizer.save(str(model_path))
            logger.info(f"[OK] Modelo guardado en: {model_path}")
        
        # Guardar metadatos
        metadata_dict = {
            'timestamp': datetime.now().isoformat(),
            'method': self.method,
            'n_samples': self.n_samples,
            'seed': self.seed,
            'shape': synthetic_df.shape,
            'columns': list(synthetic_df.columns),
            'config': self.config,
            'wealth_stage': self.wealth_report
        }

        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"[OK] Metadatos guardados en: {metadata_path}")
        
        return {
            'data': str(data_path),
            'model': str(model_path) if model_path else None,
            'metadata': str(metadata_path)
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'ClientGenerator':
        """Cargar modelo pre-entrenado."""
        logger.info(f"Cargando modelo desde: {model_path}")
        
        # Detectar método desde nombre de archivo
        if 'bootstrap' in model_path:
            generator = cls(method='bootstrap')
            generator.synthesizer = BootstrapSynthesizer.load(model_path)
            return generator
        method = 'ctgan' if 'ctgan' in model_path else 'copula'

        generator = cls(method=method)
        CTGANSynthesizer, GaussianCopulaSynthesizer, _, _ = _sdv()

        if method == 'ctgan':
            generator.synthesizer = CTGANSynthesizer.load(model_path)
        elif method == 'copula':
            generator.synthesizer = GaussianCopulaSynthesizer.load(model_path)
        
        logger.info("[OK] Modelo cargado exitosamente")
        return generator


def main():
    """Ejemplo de uso."""
    
    # Inicializar generador
    generator = ClientGenerator(
        n_samples=10000,
        method='ctgan',
        seed=42
    )
    
    # Stage 1: tabla de hogares INE (EPF) -> RiskSynth AI
    real_data = wealth_profile.build_stage1_table()
    generator._validate_data(real_data)
    
    # Entrenar
    generator.fit(real_data)
    
    # Generar sintéticos
    synthetic_data = generator.generate()
    
    # Guardar
    output_dir = str(SYNTHETIC_DIR / "clients" / f"v1_{datetime.now():%Y%m%d}")
    paths = generator.save(output_dir, synthetic_data)
    
    print(f"\n[OK] Generacion completada!")
    print(f"[DATA] Datos: {paths['data']}")
    print(f"[MODEL] Modelo: {paths['model']}")
    print(f"[META] Metadatos: {paths['metadata']}")


if __name__ == "__main__":
    main()
