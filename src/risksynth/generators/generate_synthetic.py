#!/usr/bin/env python3
# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""
Generador de datos bancarios sintéticos para España (ML).
Basado en microdatos INE + estructura WealthReader.

Uso: python -m risksynth.generators.generate_synthetic   (o: risksynth-generate)
"""

import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

# === CONFIGURACIÓN ===
from risksynth.engine import wealth_profile
from risksynth.engine.stage2 import run_stage2
from risksynth.paths import PROCESSED_DIR, SYNTHETIC_DIR

# Optional override (CI, tests): RISKSYNTH_OUTPUT_DIR=/path/to/dir
OUTPUT_DIR = Path(os.environ.get("RISKSYNTH_OUTPUT_DIR", SYNTHETIC_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Household-finance inputs carried from the (resolved) INE base row into the flat record
WEALTH_PASSTHROUGH = ['nivel_estudios', 'contract_type', 'net_income_annual_eur',
                      'household_expenses_annual_eur', 'debt_service_annual_eur',
                      'main_residence_value_eur', 'n_miembros', 'dependent_children',
                      'household_finance_source']
STAGE2_SEED_OFFSET = 1000   # model_config.yaml: pipeline.stage_2_wealth_features.seed_offset

# Datos de referencia España
CCAA_CODES = {
    'ES11': 'Galicia', 'ES12': 'Asturias', 'ES13': 'Cantabria',
    'ES21': 'País Vasco', 'ES22': 'Navarra', 'ES23': 'La Rioja',
    'ES24': 'Aragón', 'ES30': 'Madrid', 'ES41': 'Castilla y León',
    'ES42': 'Castilla-La Mancha', 'ES43': 'Extremadura', 'ES51': 'Cataluña',
    'ES52': 'Comunidad Valenciana', 'ES53': 'Baleares', 'ES61': 'Andalucía',
    'ES62': 'Murcia', 'ES63': 'Ceuta', 'ES64': 'Melilla', 'ES70': 'Canarias'
}

CIUDADES_POR_CCAA = {
    'ES30': ['Madrid', 'Alcalá de Henares', 'Móstoles', 'Getafe', 'Leganés'],
    'ES51': ['Barcelona', 'Hospitalet', 'Terrassa', 'Badalona', 'Sabadell'],
    'ES61': ['Sevilla', 'Málaga', 'Córdoba', 'Granada', 'Almería'],
    'ES52': ['Valencia', 'Alicante', 'Elche', 'Castellón', 'Torrent'],
    'ES21': ['Bilbao', 'Vitoria', 'San Sebastián', 'Barakaldo', 'Getxo'],
}

CALLES = ['Gran Vía', 'Calle Mayor', 'Paseo de la Castellana', 'Avenida Diagonal',
          'Calle Serrano', 'Ramblas', 'Calle Alcalá', 'Paseo de Gracia',
          'Calle Princesa', 'Avenida de América', 'Calle Bailén', 'Calle Goya']

NOMBRES_M = ['Antonio', 'Manuel', 'José', 'Francisco', 'David', 'Juan', 'Carlos',
             'Jesús', 'Javier', 'Daniel', 'Miguel', 'Rafael', 'Pablo', 'Pedro']
NOMBRES_F = ['María', 'Carmen', 'Ana', 'Isabel', 'Laura', 'Cristina', 'Marta',
             'Lucía', 'Elena', 'Sara', 'Paula', 'Nuria', 'Rosa', 'Pilar']
APELLIDOS = ['García', 'Rodríguez', 'Martínez', 'López', 'González', 'Hernández',
             'Pérez', 'Sánchez', 'Ramírez', 'Torres', 'Flores', 'Rivera', 'Gómez',
             'Díaz', 'Moreno', 'Muñoz', 'Álvarez', 'Romero', 'Navarro', 'Ruiz']

OCUPACIONES = ['Analista', 'Ingeniero', 'Médico', 'Profesor', 'Abogado', 'Comercial',
               'Administrativo', 'Técnico', 'Enfermero', 'Arquitecto', 'Consultor',
               'Director', 'Gerente', 'Diseñador', 'Programador', 'Business Manager']

ESTADOS_CIVILES = ['single', 'married', 'divorced', 'widowed', 'separated']

BANCOS_ES = [
    ('0049', 'Santander'), ('2100', 'CaixaBank'), ('0182', 'BBVA'),
    ('0081', 'Sabadell'), ('2085', 'Ibercaja'), ('0128', 'Bankinter'),
    ('0487', 'Unicaja'), ('3058', 'Cajamar'), ('2103', 'Unicaja')
]

CATEGORIAS_GASTO = {
    'Alimentación': ['Supermercados', 'Mercados', 'Carnicería', 'Frutería'],
    'Transporte': ['Gasolina', 'Transporte público', 'Taxi', 'Parking', 'Peajes'],
    'Vivienda': ['Alquiler', 'Hipoteca', 'Comunidad', 'Seguros hogar'],
    'Suministros': ['Electricidad', 'Gas', 'Agua', 'Internet', 'Teléfono'],
    'Ocio': ['Restaurantes', 'Cine', 'Viajes', 'Deportes', 'Suscripciones'],
    'Salud': ['Farmacia', 'Médico', 'Dentista', 'Óptica'],
    'Ropa': ['Tiendas ropa', 'Calzado', 'Complementos'],
    'Educación': ['Colegios', 'Cursos', 'Libros', 'Material'],
    'Transferencias': ['Bizum enviado', 'Transferencia emitida'],
    'Otros': ['Varios', 'Otros gastos']
}

CATEGORIAS_INGRESO = {
    'Nómina': ['Salario', 'Paga extra', 'Bonus'],
    'Transferencias': ['Bizum recibido', 'Transferencia recibida'],
    'Otros ingresos': ['Devolución', 'Reembolso', 'Intereses']
}

TIPOS_CONTRATO = ['card', 'insurance', 'loan', 'account']
SUBTIPOS_SEGURO = ['life', 'auto', 'home', 'health']
TIPOS_INVERSION = ['funds', 'stocks', 'pension_plan', 'etf']

EMPRESAS = [
    'Tech Innovations S.L.', 'Global Dynamics S.A.', 'Data Enterprises S.L.',
    'Grupo Sistemas S.A.', 'Soluciones Digitales S.L.', 'Consultoría Avanzada S.A.',
    'Industrias Unidas S.A.', 'Servicios Integrales S.L.', 'Desarrollo Plus S.A.'
]


# === MODELO DE DEFAULT v7.0 - LOGÍSTICO CALIBRADO ===
# Coeficientes basados en literatura BdE, EFF 2022, estudios morosidad España
# Referencias:
#   - Banco de España: Informe de Estabilidad Financiera (2023-2024)
#   - EFF 2022: Encuesta Financiera de las Familias
#   - Estudios académicos morosidad hipotecaria/consumo España

# Tasas base por segmento (fuente: BdE, EFF 2022)
TASAS_SEGMENTO = {
    'edad': {
        (0, 30): 0.080,    # Jóvenes: 8% (EFF: mayor riesgo relativo)
        (30, 45): 0.050,   # Adultos: 5%
        (45, 55): 0.040,   # Maduros: 4%
        (55, 100): 0.030,  # Mayores: 3% (ingresos estables)
    },
    'renta': {
        (0, 15000): 0.120,      # Bajo: 12% (ECV: alta correlación pobreza-morosidad)
        (15000, 25000): 0.060,  # Medio-bajo: 6%
        (25000, 40000): 0.035,  # Medio: 3.5%
        (40000, float('inf')): 0.020,  # Alto: 2%
    },
    'producto': {
        'hipoteca': 0.025,      # BdE 2024: 2.5% morosidad hipotecaria
        'consumo': 0.055,       # BdE 2024: 5.5% morosidad consumo
        'ambos': 0.085,         # Doble carga: ~8.5%
        'ninguno': 0.030,       # Sin deuda formal: 3% (otros impagos)
    }
}

# Coeficientes logísticos (log-odds) calibrados
# Fuente: meta-análisis estudios morosidad España + EFF
COEF_LOGIT = {
    'intercepto': -4.0,              # Tasa base ~1.8% para perfil óptimo
    'ratio_endeudamiento': 3.5,      # OR=33 por unidad (muy significativo)
    'log_ingresos': -0.45,           # OR=0.64 por log-unidad
    'log_balance': -0.25,            # OR=0.78 por log-unidad  
    'tasa_ahorro_neg': 1.2,          # OR=3.3 si ahorro negativo
    'edad_joven': 0.6,               # OR=1.8 si <30
    'edad_mayor': -0.3,              # OR=0.74 si >55
    'tiene_hipoteca': 0.4,           # OR=1.5
    'tiene_prestamo': 0.7,           # OR=2.0
    'doble_deuda': 0.5,              # OR=1.6 (interacción)
    'ccaa_arope_alto': 0.5,          # OR=1.6 (Andalucía, Extremadura, CLM)
    'ratio_gastos_alto': 0.8,        # OR=2.2 si gastos/ingresos > 0.9
    'sin_colchon': 0.9,              # OR=2.5 si balance < 1 mes gastos
    'joven_pobre': 0.7,              # OR=2.0 (interacción edad×renta)
    'ocio_excesivo': 0.4,            # OR=1.5 si ocio > 15% gastos
}


def calcular_prob_default_logit(record):
    """
    Calcula P(default) usando modelo logístico calibrado.
    Retorna probabilidad entre 0 y 1.
    """
    # Extraer variables
    ratio = record.get('ratio_endeudamiento', 0)
    ingresos = max(record.get('ingresos_anuales', 30000), 1000)
    balance = max(record.get('balance_cuentas', 5000), 100)
    tasa_ahorro = record.get('tasa_ahorro', 0.15)
    edad = record.get('edad', 40)
    gastos = record.get('gastos_anuales', 20000)
    gasto_ocio = record.get('gasto_ocio', 0)
    tiene_hipoteca = record.get('tiene_hipoteca', 0)
    tiene_prestamo = record.get('tiene_prestamo_personal', 0)
    cp = str(record.get('codigo_postal', ''))
    
    # Calcular log-odds
    log_odds = COEF_LOGIT['intercepto']
    
    # Efecto principal: ratio endeudamiento (no lineal via logit)
    log_odds += COEF_LOGIT['ratio_endeudamiento'] * ratio
    
    # Efecto log-lineal: ingresos y balance
    log_odds += COEF_LOGIT['log_ingresos'] * np.log(ingresos / 10000)
    log_odds += COEF_LOGIT['log_balance'] * np.log(balance / 1000)
    
    # Efectos binarios
    if tasa_ahorro < 0:
        log_odds += COEF_LOGIT['tasa_ahorro_neg']
    
    if edad < 30:
        log_odds += COEF_LOGIT['edad_joven']
    elif edad > 55:
        log_odds += COEF_LOGIT['edad_mayor']
    
    if tiene_hipoteca:
        log_odds += COEF_LOGIT['tiene_hipoteca']
    
    if tiene_prestamo:
        log_odds += COEF_LOGIT['tiene_prestamo']
    
    # Interacción: doble deuda
    if tiene_hipoteca and tiene_prestamo:
        log_odds += COEF_LOGIT['doble_deuda']
    
    # Efecto regional (CCAA alto AROPE)
    ccaa_riesgo = ('41', '14', '29', '18', '04', '21', '11', '23',  # Andalucía
                   '06', '10',  # Extremadura
                   '02', '13', '16', '19', '45')  # C-La Mancha
    if cp[:2] in ccaa_riesgo:
        log_odds += COEF_LOGIT['ccaa_arope_alto']
    
    # Ratio gastos/ingresos alto
    if ingresos > 0 and (gastos / ingresos) > 0.9:
        log_odds += COEF_LOGIT['ratio_gastos_alto']
    
    # Sin colchón financiero (< 1 mes de gastos)
    gastos_mes = gastos / 12
    if balance < gastos_mes:
        log_odds += COEF_LOGIT['sin_colchon']
    
    # Interacción: joven + bajos ingresos
    if edad < 30 and ingresos < 20000:
        log_odds += COEF_LOGIT['joven_pobre']
    
    # Ocio excesivo (> 15% de gastos)
    if gastos > 0 and (gasto_ocio / gastos) > 0.15:
        log_odds += COEF_LOGIT['ocio_excesivo']
    
    # Convertir a probabilidad (sigmoide)
    prob = 1 / (1 + np.exp(-log_odds))
    
    return prob


def get_tasa_base_segmento(record):
    """
    Obtiene tasa base estratificada por segmento demográfico.
    Combina edad, renta y producto.
    """
    edad = record.get('edad', 40)
    ingresos = record.get('ingresos_anuales', 30000)
    tiene_hipoteca = record.get('tiene_hipoteca', 0)
    tiene_prestamo = record.get('tiene_prestamo_personal', 0)
    
    # Tasa por edad
    tasa_edad = 0.05
    for (min_e, max_e), tasa in TASAS_SEGMENTO['edad'].items():
        if min_e <= edad < max_e:
            tasa_edad = tasa
            break
    
    # Tasa por renta
    tasa_renta = 0.05
    for (min_r, max_r), tasa in TASAS_SEGMENTO['renta'].items():
        if min_r <= ingresos < max_r:
            tasa_renta = tasa
            break
    
    # Tasa por producto
    if tiene_hipoteca and tiene_prestamo:
        tasa_producto = TASAS_SEGMENTO['producto']['ambos']
    elif tiene_hipoteca:
        tasa_producto = TASAS_SEGMENTO['producto']['hipoteca']
    elif tiene_prestamo:
        tasa_producto = TASAS_SEGMENTO['producto']['consumo']
    else:
        tasa_producto = TASAS_SEGMENTO['producto']['ninguno']
    
    # Promedio ponderado (edad: 30%, renta: 40%, producto: 30%)
    tasa_base = 0.30 * tasa_edad + 0.40 * tasa_renta + 0.30 * tasa_producto
    
    return tasa_base


def calcular_score_riesgo(record):
    """
    Calcula score de riesgo combinando modelo logístico + tasas segmento.
    Retorna score normalizado 0-1.
    """
    # Probabilidad del modelo logístico
    prob_logit = calcular_prob_default_logit(record)
    
    # Tasa base del segmento
    tasa_segmento = get_tasa_base_segmento(record)
    
    # Combinar: 70% logit + 30% segmento (el logit captura más matices)
    prob_combinada = 0.70 * prob_logit + 0.30 * tasa_segmento
    
    # Añadir ruido (factores no observados)
    ruido = random.gauss(0, 0.015)
    prob_final = prob_combinada + ruido
    
    # Limitar a [0.01, 0.95]
    return max(0.01, min(prob_final, 0.95))


def determinar_default(record, target_rate=None):
    """
    Determina default (0/1) basado en probabilidad calibrada.
    
    target_rate: Si se especifica, ajusta para alcanzar tasa global aproximada.
    """
    prob = calcular_score_riesgo(record)
    
    if target_rate is not None:
        # Ajustar probabilidad para alcanzar tasa objetivo
        # Usamos factor de escala basado en tasa esperada vs objetivo
        tasa_esperada = 0.06  # ~6% tasa media del modelo
        factor = target_rate / tasa_esperada
        prob = prob * factor
        prob = max(0.005, min(prob, 0.90))
    
    return 1 if random.random() < prob else 0, prob


# === FUNCIONES AUXILIARES ===

def gen_dni():
    """Genera DNI español válido."""
    letras = 'TRWAGMYFPDXBNJZSQVHLCKE'
    num = random.randint(10000000, 99999999)
    return f"{num}{letras[num % 23]}"

def gen_iban(banco_code=None):
    """Genera IBAN español."""
    if not banco_code:
        banco_code = random.choice(BANCOS_ES)[0]
    cuenta = ''.join([str(random.randint(0, 9)) for _ in range(16)])
    # Simplificado - en producción calcular dígitos control
    dc = str(random.randint(10, 99))
    return f"ES{dc}{banco_code}{cuenta}"

def gen_email(nombre, apellido):
    """Genera email basado en nombre."""
    dominos = ['gmail.com', 'hotmail.com', 'yahoo.es', 'outlook.com', 'icloud.com']
    nombre_clean = nombre.lower().replace(' ', '').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    apellido_clean = apellido.lower().replace(' ', '').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    variante = random.choice([
        f"{nombre_clean}.{apellido_clean}",
        f"{nombre_clean}{apellido_clean}",
        f"{nombre_clean[0]}{apellido_clean}",
        f"{nombre_clean}{random.randint(1, 99)}"
    ])
    return f"{variante}@{random.choice(dominos)}"

def gen_telefono():
    """Genera móvil español."""
    prefijos = ['6', '7']
    return f"+34{random.choice(prefijos)}{random.randint(10000000, 99999999)}"

def gen_fecha_nacimiento(edad):
    """Genera fecha nacimiento dada edad."""
    hoy = datetime.now()
    año = hoy.year - edad
    mes = random.randint(1, 12)
    dia = random.randint(1, 28)
    return f"{año}-{mes:02d}-{dia:02d}"

def gen_direccion():
    """Genera dirección aleatoria."""
    return f"{random.choice(CALLES)} {random.randint(1, 150)}"

def gen_codigo_postal(ccaa):
    """Genera CP basado en CCAA."""
    prefijos_cp = {
        'ES30': '28', 'ES51': '08', 'ES61': '41', 'ES52': '46',
        'ES21': '48', 'ES11': '15', 'ES24': '50', 'ES53': '07'
    }
    prefijo = prefijos_cp.get(ccaa, str(random.randint(10, 52)))
    return f"{prefijo}{random.randint(0, 9):01d}{random.randint(0, 99):02d}"

def rand_date(start_days_ago, end_days_ago=0):
    """Fecha aleatoria entre hace X días y hace Y días."""
    hoy = datetime.now()
    delta = random.randint(end_days_ago, start_days_ago)
    fecha = hoy - timedelta(days=delta)
    return fecha.strftime('%Y-%m-%d')


# === GENERADORES DE SECCIONES ===

def gen_user_information(row):
    """Genera user_information basado en fila de datos."""
    sexo = row.get('sexo', random.choice([1, 2]))
    es_mujer = sexo == 2 or str(sexo).lower() in ['female', 'mujer', 'f', '2']
    
    nombre = random.choice(NOMBRES_F if es_mujer else NOMBRES_M)
    apellido1 = random.choice(APELLIDOS)
    apellido2 = random.choice(APELLIDOS)
    nombre_completo = f"{nombre} {apellido1} {apellido2}"
    
    edad = row.get('edad', None)
    # Manejar NaN, None, strings
    if edad is None or (isinstance(edad, float) and np.isnan(edad)):
        edad = random.randint(25, 65)
    elif isinstance(edad, str):
        try:
            edad = int(float(edad))
        except:
            edad = random.randint(25, 65)
    else:
        try:
            edad = int(edad)
        except:
            edad = random.randint(25, 65)
    edad = max(18, min(80, edad))
    
    ccaa = row.get('ccaa', random.choice(list(CCAA_CODES.keys())))
    if ccaa not in CCAA_CODES:
        ccaa = random.choice(list(CCAA_CODES.keys()))
    
    ciudades = CIUDADES_POR_CCAA.get(ccaa, [CCAA_CODES.get(ccaa, 'Madrid')])
    ciudad = random.choice(ciudades)
    
    return {
        'ID': gen_dni(),
        'name': nombre_completo,
        'email': gen_email(nombre, apellido1),
        'cell_phone': gen_telefono(),
        'gender': 'female' if es_mujer else 'male',
        'birth_date': gen_fecha_nacimiento(edad),
        'address': gen_direccion(),
        'city': ciudad,
        'country': 'ES',
        'postal_code': gen_codigo_postal(ccaa),
        'birth_place': 'ES',
        'marital_status': random.choice(ESTADOS_CIVILES),
        'job': row.get('ocupacion', random.choice(OCUPACIONES))
    }


def gen_transactions(n_meses=12, ingreso_mensual=2500, perfil_gasto=None, gasto_mensual=None):
    """Genera transacciones realistas."""
    transactions = []
    hoy = datetime.now()
    
    # Perfil de gasto por defecto basado en EPF
    if perfil_gasto is None:
        perfil_gasto = {
            'Vivienda': 0.32, 'Alimentación': 0.16, 'Transporte': 0.11,
            'Suministros': 0.08, 'Ocio': 0.09, 'Salud': 0.05,
            'Ropa': 0.05, 'Educación': 0.03, 'Otros': 0.06
        }
    
    if gasto_mensual is None:
        gasto_mensual = ingreso_mensual * 0.75  # Ahorro ~25% (sin datos EPF del hogar)
    balance = random.uniform(5000, 50000)
    
    for mes in range(n_meses):
        fecha_base = hoy - timedelta(days=30 * mes)
        
        # Nómina (día 28-30 del mes anterior o 1-5 del actual)
        dia_nomina = random.randint(28, 30) if random.random() > 0.5 else random.randint(1, 5)
        fecha_nomina = fecha_base.replace(day=min(dia_nomina, 28))
        variacion_salario = random.uniform(0.95, 1.05)
        salario = round(ingreso_mensual * variacion_salario, 2)
        balance += salario
        
        transactions.append({
            'date': fecha_nomina.strftime('%Y-%m-%d'),
            'amount': salario,
            'balance': round(balance, 2),
            'description': f"NOMINA {fecha_nomina.strftime('%m/%Y')}",
            'category': 'Nómina',
            'subcategory': 'Salario'
        })
        
        # Gastos del mes
        for categoria, porcentaje in perfil_gasto.items():
            if categoria not in CATEGORIAS_GASTO:
                continue
                
            gasto_cat = gasto_mensual * porcentaje
            
            # Dividir en varias transacciones
            if categoria in ['Vivienda', 'Suministros']:
                n_trans = 1  # Pagos únicos mensuales
            elif categoria == 'Alimentación':
                n_trans = random.randint(8, 15)  # Compras frecuentes
            else:
                n_trans = random.randint(2, 6)
            
            for _ in range(n_trans):
                monto = round(gasto_cat / n_trans * random.uniform(0.7, 1.3), 2)
                dia = random.randint(1, 28)
                fecha_gasto = fecha_base.replace(day=dia)
                balance -= monto
                
                subcats = CATEGORIAS_GASTO.get(categoria, ['Otros'])
                
                transactions.append({
                    'date': fecha_gasto.strftime('%Y-%m-%d'),
                    'amount': -monto,
                    'balance': round(balance, 2),
                    'description': f"{random.choice(subcats).upper()}",
                    'category': categoria,
                    'subcategory': random.choice(subcats)
                })
        
        # Bizum aleatorios
        for _ in range(random.randint(0, 5)):
            es_ingreso = random.random() > 0.6
            monto = round(random.uniform(5, 100), 2)
            dia = random.randint(1, 28)
            fecha_bizum = fecha_base.replace(day=dia)
            
            if es_ingreso:
                balance += monto
                cat, subcat = 'Transferencias', 'Bizum recibido'
            else:
                monto = -monto
                balance -= abs(monto)
                cat, subcat = 'Transferencias', 'Bizum enviado'
            
            transactions.append({
                'date': fecha_bizum.strftime('%Y-%m-%d'),
                'amount': monto,
                'balance': round(balance, 2),
                'description': f"BIZUM {'DE' if es_ingreso else 'A'} CONTACTO",
                'category': cat,
                'subcategory': subcat
            })
    
    # Ordenar por fecha descendente
    transactions.sort(key=lambda x: x['date'], reverse=True)
    return transactions, balance


def gen_accounts(ingreso_mensual, n_cuentas=None, gasto_mensual=None, mortgage_monthly=None):
    """Genera cuentas bancarias.

    gasto_mensual: gasto mensual del hogar (EPF); None = 75% del ingreso (legacy).
    mortgage_monthly: cuota hipotecaria mensual del hogar (EPF CAPITMENS + INTERESMENS).
        None = hipoteca aleatoria (30%, legacy); 0 = sin hipoteca; > 0 = hipoteca con esa cuota.
    """
    if n_cuentas is None:
        n_cuentas = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    
    accounts = []
    banco_principal = random.choice(BANCOS_ES)
    
    for i in range(n_cuentas):
        banco = banco_principal if i == 0 else random.choice(BANCOS_ES)
        tipo = 'checking' if i == 0 else random.choice(['checking', 'savings'])
        
        # Generar transacciones solo para cuenta principal
        if i == 0:
            trans, balance = gen_transactions(n_meses=12, ingreso_mensual=ingreso_mensual,
                                              gasto_mensual=gasto_mensual)
        else:
            trans = []
            balance = random.uniform(1000, 30000) if tipo == 'savings' else random.uniform(500, 5000)
        
        # Calcular agregaciones
        ingresos_total = sum(t['amount'] for t in trans if t['amount'] > 0)
        gastos_total = abs(sum(t['amount'] for t in trans if t['amount'] < 0))
        
        gastos_por_cat = {}
        for t in trans:
            if t['amount'] < 0:
                cat = t['category']
                gastos_por_cat[cat] = gastos_por_cat.get(cat, 0) + abs(t['amount'])
        
        # Préstamos (probabilidad según ingreso)
        loans = []
        if mortgage_monthly is None:
            tiene_hipoteca = random.random() < 0.3      # legacy: 30% aleatorio
        else:
            tiene_hipoteca = mortgage_monthly > 0      # alineado con la deuda EPF del hogar
        if tiene_hipoteca:
            años_rest = random.randint(5, 25)
            tae = random.uniform(1.5, 4.5)
            if mortgage_monthly:
                # Cuota observada (EPF) -> capital pendiente con la misma fórmula de cuota
                cuota = round(mortgage_monthly, 2)
                pendiente = cuota * años_rest * 12 / (1 + tae / 100)
                principal = pendiente / random.uniform(0.3, 0.9)
            else:
                principal = random.uniform(100000, 400000)
                pendiente = principal * random.uniform(0.3, 0.9)
                cuota = round(pendiente / (años_rest * 12) * (1 + tae/100), 2)

            loans.append({
                'type': 'mortgage',
                'subtype': 'fixed',
                'original_amount': round(principal, 2),
                'pending_amount': round(pendiente, 2),
                'monthly_payment': cuota,
                'interest_rate': round(tae, 2),
                'start_date': rand_date(3650, 365),  # 1-10 años atrás
                'end_date': rand_date(-365, -años_rest*365)  # Futuro
            })
        
        if random.random() < 0.15:  # 15% tiene préstamo personal
            principal = random.uniform(3000, 30000)
            pendiente = principal * random.uniform(0.2, 0.8)
            loans.append({
                'type': 'personal',
                'subtype': 'consumer',
                'original_amount': round(principal, 2),
                'pending_amount': round(pendiente, 2),
                'monthly_payment': round(pendiente / random.randint(12, 48), 2),
                'interest_rate': round(random.uniform(5, 12), 2),
                'start_date': rand_date(1095, 180),
                'end_date': rand_date(-180, -1460)
            })
        
        account = {
            'type': tipo,
            'IBAN': gen_iban(banco[0]),
            'balance': round(balance, 2),
            'currency': 'EUR',
            'bank_name': banco[1],
            'holders': [{'name': 'TITULAR', 'role': 'owner'}],
            'transactions': trans,
            'loans': loans if i == 0 else [],
            'aggregations': {
                'income': {
                    'total': round(ingresos_total, 2),
                    'salary': round(ingresos_total * 0.9, 2),
                    'other': round(ingresos_total * 0.1, 2)
                },
                'expenses': {
                    'total': round(gastos_total, 2),
                    **{k: round(v, 2) for k, v in gastos_por_cat.items()}
                }
            }
        }
        accounts.append(account)
    
    return accounts


def gen_contracts(n_años_historial=3):
    """Genera contratos (tarjetas, seguros)."""
    contracts = []
    
    # Tarjetas (1-4)
    n_tarjetas = random.randint(1, 4)
    for _ in range(n_tarjetas):
        contracts.append({
            'signing_date': rand_date(n_años_historial * 365, 30),
            'type': 'card',
            'subtype': random.choice(['debit', 'credit']),
            'description': 'TARJETA ***'
        })
    
    # Seguros (0-3)
    n_seguros = random.randint(0, 3)
    for _ in range(n_seguros):
        contracts.append({
            'signing_date': rand_date(n_años_historial * 365, 30),
            'type': 'insurance',
            'subtype': random.choice(SUBTIPOS_SEGURO),
            'description': 'SEGURO ***'
        })
    
    return contracts


def gen_investments(patrimonio_financiero=None):
    """Genera inversiones."""
    if patrimonio_financiero is None:
        # 50% no tiene inversiones
        if random.random() < 0.5:
            return []
        patrimonio_financiero = random.uniform(5000, 200000)
    
    investments = []
    tipos = random.sample(TIPOS_INVERSION, k=random.randint(1, 3))
    
    for tipo in tipos:
        porcentaje = random.uniform(0.2, 0.6)
        monto = patrimonio_financiero * porcentaje
        
        inv = {
            'type': tipo,
            'subtype': tipo,
            'amount': round(monto, 2),
            'currency': 'EUR',
            'entity': random.choice(BANCOS_ES)[1],
            'positions': []
        }
        
        # Posiciones dentro de la inversión
        n_pos = random.randint(1, 5)
        for _ in range(n_pos):
            pos_monto = monto / n_pos * random.uniform(0.5, 1.5)
            precio_compra = random.uniform(10, 500)
            variacion = random.uniform(-0.2, 0.4)
            
            inv['positions'].append({
                'name': f"{'Fondo' if tipo == 'funds' else 'Acción'} {random.randint(1000, 9999)}",
                'isin': f"ES{random.randint(1000000000, 9999999999)}",
                'quantity': round(pos_monto / precio_compra, 4),
                'purchase_price': round(precio_compra, 2),
                'current_price': round(precio_compra * (1 + variacion), 2),
                'currency': 'EUR'
            })
        
        investments.append(inv)
    
    return investments


def gen_employers():
    """Genera empleadores detectados."""
    n_emp = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
    employers = []
    
    for _ in range(n_emp):
        employers.append({
            'uuid': uuid.uuid4().hex,
            'company_id': gen_dni(),  # CIF simplificado
            'company_name': random.choice(EMPRESAS)
        })
    
    return employers


# === GENERADOR PRINCIPAL ===

def _num(value):
    """Float or None for missing/NaN values coming from base rows."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else value


def generate_full_payload(row):
    """Genera payload JSON completo estilo WealthReader."""
    # Estimar ingreso mensual
    # Household income: HY020 if observed, else the (EPF or donor) household net income,
    # else ECV gross employee income. Keeps the payload coherent with the wealth profile.
    renta = None
    for column in ('renta_disponible', 'net_income_annual_eur', 'ingresos_trabajo'):
        value = row.get(column)
        if value is not None and not pd.isna(value):
            renta = value
            break
    if renta and not pd.isna(renta):
        try:
            ingreso_mensual = float(renta) / 12
        except:
            ingreso_mensual = random.uniform(1500, 4000)
    else:
        ingreso_mensual = random.uniform(1500, 4000)
    
    # Household finances from the resolved INE row (stage 2 inputs), when present
    gastos_hogar = _num(row.get('household_expenses_annual_eur'))
    deuda_hogar = _num(row.get('debt_service_annual_eur'))
    if gastos_hogar is not None:
        # Aligned with EPF: no artificial income floor/cap, spending = EPF household spending
        ingreso_mensual = max(100.0, ingreso_mensual)
        gasto_mensual = gastos_hogar / 12
        mortgage_monthly = (deuda_hogar or 0.0) / 12
    else:
        ingreso_mensual = max(1000, min(15000, ingreso_mensual))
        gasto_mensual, mortgage_monthly = None, None

    user_info = gen_user_information(row)
    accounts = gen_accounts(ingreso_mensual, gasto_mensual=gasto_mensual,
                            mortgage_monthly=mortgage_monthly)
    
    payload = {
        'success': True,
        'payload': {
            'user_information': user_info,
            'contracts': gen_contracts(),
            'accounts': accounts,
            'investments': gen_investments(),
            'employers': gen_employers(),
            'files': []
        },
        'statistics': {
            'code': 'synthetic',
            'execution_time': round(random.uniform(0.01, 0.1), 4),
            'operation_id': uuid.uuid4().hex[:20]
        }
    }
    
    return payload


def generate_flat_record(row, target_default_rate=None):
    """Genera registro plano para ML con features de interacción y target calibrado."""
    payload = generate_full_payload(row)
    user = payload['payload']['user_information']
    accounts = payload['payload']['accounts']
    
    # Cuenta principal
    cuenta_ppal = accounts[0] if accounts else {}
    agg = cuenta_ppal.get('aggregations', {})
    loans = cuenta_ppal.get('loans', [])
    
    # Calcular métricas derivadas
    ingresos = agg.get('income', {}).get('total', 0)
    gastos = agg.get('expenses', {}).get('total', 0)
    balance = cuenta_ppal.get('balance', 0)
    
    deuda_total = sum(l.get('pending_amount', 0) for l in loans)
    cuota_total = sum(l.get('monthly_payment', 0) for l in loans)
    cuota_personal = sum(l.get('monthly_payment', 0) for l in loans if l['type'] == 'personal')
    # Liquid assets: positive balances only (an overdrawn account is not a liquid asset)
    saldo_total = sum(max(a.get('balance', 0), 0) for a in accounts)

    # Ratio endeudamiento
    ingreso_mensual = ingresos / 12 if ingresos > 0 else 1
    ratio_endeudamiento = cuota_total / ingreso_mensual if ingreso_mensual > 0 else 0
    
    # Edad desde fecha nacimiento
    try:
        birth = datetime.strptime(user['birth_date'], '%Y-%m-%d')
        edad = (datetime.now() - birth).days // 365
    except:
        edad = row.get('edad', 35)
    
    # Tasa ahorro
    tasa_ahorro = (ingresos - gastos) / ingresos if ingresos > 0 else 0
    
    # Gastos por categoría
    gasto_vivienda = agg.get('expenses', {}).get('Vivienda', 0)
    gasto_alimentacion = agg.get('expenses', {}).get('Alimentación', 0)
    gasto_transporte = agg.get('expenses', {}).get('Transporte', 0)
    gasto_ocio = agg.get('expenses', {}).get('Ocio', 0)
    gasto_suministros = agg.get('expenses', {}).get('Suministros', 0)
    
    # Flags binarios
    tiene_hipoteca = 1 if any(l['type'] == 'mortgage' for l in loans) else 0
    tiene_prestamo = 1 if any(l['type'] == 'personal' for l in loans) else 0
    tiene_inversiones = 1 if payload['payload']['investments'] else 0
    
    record = {
        # Identificador
        'id': user['ID'],
        
        # === DEMOGRÁFICOS ===
        'edad': edad,
        'sexo': 1 if user['gender'] == 'male' else 2,
        'estado_civil': user['marital_status'],
        'ciudad': user['city'],
        'codigo_postal': user['postal_code'],
        'ocupacion': user['job'],
        
        # === INGRESOS Y GASTOS ===
        'ingresos_anuales': round(ingresos, 2),
        'gastos_anuales': round(gastos, 2),
        'ahorro_anual': round(ingresos - gastos, 2),
        'tasa_ahorro': round(tasa_ahorro, 4),
        
        # === GASTOS POR CATEGORÍA ===
        'gasto_vivienda': round(gasto_vivienda, 2),
        'gasto_alimentacion': round(gasto_alimentacion, 2),
        'gasto_transporte': round(gasto_transporte, 2),
        'gasto_ocio': round(gasto_ocio, 2),
        'gasto_suministros': round(gasto_suministros, 2),
        
        # === PATRIMONIO ===
        'balance_cuentas': round(balance, 2),
        'n_cuentas': len(accounts),
        'tiene_inversiones': tiene_inversiones,
        'valor_inversiones': sum(i.get('amount', 0) for i in payload['payload']['investments']),
        
        # === DEUDA ===
        'tiene_hipoteca': tiene_hipoteca,
        'tiene_prestamo_personal': tiene_prestamo,
        'deuda_total': round(deuda_total, 2),
        'cuota_mensual_total': round(cuota_total, 2),
        'ratio_endeudamiento': round(ratio_endeudamiento, 4),
        
        # === PRODUCTOS ===
        'n_tarjetas': len([c for c in payload['payload']['contracts'] if c['type'] == 'card']),
        'n_seguros': len([c for c in payload['payload']['contracts'] if c['type'] == 'insurance']),
        
        # === ACTIVIDAD ===
        'n_transacciones_12m': len(cuenta_ppal.get('transactions', [])),
        'n_empleadores': len(payload['payload']['employers']),
        
        # === FEATURES DE INTERACCIÓN (EXPLÍCITAS) ===
        # Ratios financieros
        'ratio_gastos_ingresos': round(gastos / max(ingresos, 1), 4),
        'carga_deuda_mensual': round(cuota_total / max(ingreso_mensual, 1), 4),
        'colchon_meses': round(balance / max(gastos / 12, 1), 2),
        
        # Interacciones binarias
        'deuda_doble': tiene_hipoteca * tiene_prestamo,
        'joven_bajo_ingreso': int(edad < 30 and ingresos < 20000),
        'mayor_estable': int(edad > 50 and balance > 20000),
        'sin_colchon': int(balance < (gastos / 12)),
        
        # Ratios de comportamiento
        'ocio_sobre_gastos': round(gasto_ocio / max(gastos, 1), 4),
        'vivienda_sobre_ingresos': round(gasto_vivienda / max(ingresos, 1), 4),
        'esenciales_sobre_gastos': round((gasto_vivienda + gasto_alimentacion) / max(gastos, 1), 4),
        
        # Indicadores compuestos
        'stress_financiero': round(
            (cuota_total + gasto_vivienda/12) / max(ingreso_mensual, 1), 4
        ),
        'capacidad_ahorro_real': round(
            (ingresos - gastos - cuota_total*12) / max(ingresos, 1), 4
        ),
        
        # === METADATA ===
        'situacion_laboral': row.get('situacion_laboral', None),
        'fuente_base': row.get('fuente', 'synthetic'),

        # === INPUTS RiskSynth (stage 2) ===
        'liquid_savings_eur': round(saldo_total, 2),                 # positive account balances
        'saldo_total_cuentas': round(saldo_total, 2),                # as generated (before stage 3)
        'consumer_debt_service_annual_eur': round(12 * cuota_personal, 2),
        **{k: row.get(k) for k in WEALTH_PASSTHROUGH},
    }
    
    # === TARGET: Default calibrado ===
    default_val, prob = determinar_default(record, target_default_rate)
    record['prob_default'] = round(prob, 4)
    record['default'] = default_val
    
    return record


def load_base_data():
    """Carga datos base procesados."""
    combined = PROCESSED_DIR / "combined_base.parquet"
    
    if combined.exists():
        df = pd.read_parquet(combined)
        print(f"[OK] Cargados {len(df)} registros base de combined_base.parquet")
        return df
    
    # Intentar cargar individuales
    dfs = []
    for f in ['ecv_processed.parquet', 'epf_processed.parquet', 'epa_processed.parquet']:
        path = PROCESSED_DIR / f
        if path.exists():
            dfs.append(pd.read_parquet(path))
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        print(f"[OK] Cargados {len(df)} registros de archivos individuales")
        return df
    
    print("[WARN] No hay datos base. Generando desde cero.")
    return None


def _init_worker(seed):
    """Inicializa semilla random en cada worker."""
    random.seed(seed)
    np.random.seed(seed)


def _generate_flat_worker(args):
    """Worker para generar registro plano."""
    row, seed, target_default_rate = args
    random.seed(seed)
    np.random.seed(seed)
    return generate_flat_record(row, target_default_rate)


def reconcile_credit_fields(df, target_default_rate=None, seed=42):
    """Align account-derived credit columns with the final wealth columns.

    Stage 3 can lower liquid_savings_eur (EC01/EC02 injection). For those rows the account
    balances are rescaled to the new liquid total and every balance-derived credit feature
    (balance_cuentas, colchon_meses, sin_colchon, mayor_estable) and the default target are
    recomputed. Also checks that the loan book matches the wealth debt service
    (12 * cuota_mensual_total == EPF mortgage + generator personal loans).
    """
    df = df.copy()
    stats: dict = {"rows_rebalanced": 0, "debt_mismatch_rows": None}
    if 'liquid_savings_eur' not in df or 'saldo_total_cuentas' not in df:
        return df, stats

    changed = (df['liquid_savings_eur'] - df['saldo_total_cuentas']).abs() > 0.5
    df['credit_fields_reconciled'] = changed
    if changed.any():
        factor = (df.loc[changed, 'liquid_savings_eur'] / df.loc[changed, 'saldo_total_cuentas']
                  .where(df.loc[changed, 'saldo_total_cuentas'] > 0)).fillna(0.0)
        positive = df.loc[changed, 'balance_cuentas'] > 0
        new_balance = df.loc[changed, 'balance_cuentas'].where(~positive,
                                                               df.loc[changed, 'balance_cuentas'] * factor)
        df.loc[changed, 'balance_cuentas'] = new_balance.round(2)
        df.loc[changed, 'saldo_total_cuentas'] = df.loc[changed, 'liquid_savings_eur']
        gastos_mes = (df.loc[changed, 'gastos_anuales'] / 12).clip(lower=1)
        df.loc[changed, 'colchon_meses'] = (df.loc[changed, 'balance_cuentas'] / gastos_mes).round(2)
        df.loc[changed, 'sin_colchon'] = (df.loc[changed, 'balance_cuentas'] < gastos_mes).astype(int)
        df.loc[changed, 'mayor_estable'] = ((df.loc[changed, 'edad'] > 50)
                                            & (df.loc[changed, 'balance_cuentas'] > 20000)).astype(int)
        # Default target depends on balance and sin_colchon: re-draw it deterministically per row
        for idx in df.index[changed]:
            random.seed(seed * 1_000_003 + int(idx))
            default_val, prob = determinar_default(df.loc[idx].to_dict(), target_default_rate)
            df.at[idx, 'prob_default'] = round(prob, 4)
            df.at[idx, 'default'] = default_val
        stats["rows_rebalanced"] = int(changed.sum())

    if {'debt_service_annual_eur', 'consumer_debt_service_annual_eur', 'cuota_mensual_total'} <= set(df):
        has_epf = df['debt_service_annual_eur'].notna()
        expected = df['debt_service_annual_eur'].fillna(0) + df['consumer_debt_service_annual_eur'].fillna(0)
        mismatch = has_epf & ((12 * df['cuota_mensual_total'] - expected).abs() > 12)   # > 1 EUR/month
        stats["debt_mismatch_rows"] = int(mismatch.sum())
        stats["mortgage_share_pct"] = round(100 * float(df['tiene_hipoteca'].mean()), 2)
        stats["mortgage_matches_epf_pct"] = round(100 * float(
            (df.loc[has_epf, 'tiene_hipoteca'] == (df.loc[has_epf, 'debt_service_annual_eur'] > 0)).mean()), 2)
    return df, stats


def generate_synthetic_data(n_records, base_df=None, n_jobs=None, batch_size=5000, target_default_rate=None,
                            wealth_stage=True, seed=42, wealth_cfg=None, edge_case_share=None):
    """Genera datos sintéticos para ML con paralelización y batches.

    wealth_stage: RiskSynth AI stages 2+3. Antes de generar, cada fila base recibe un bloque de
    finanzas del hogar (EPF observado o donante EPF), que fija gasto e hipoteca del payload;
    después se derivan las features wealth, se aplican CR01-CR07 + MiFID II y la cohorte de
    edge cases, y se reconcilian las columnas de crédito con los saldos finales.
    wealth_cfg: data_config ya cargado (p. ej. con un risk mix distinto); None = configs/.
    edge_case_share: override del 8% total de edge cases.
    n_jobs=1 genera en el proceso actual (sin multiprocessing).
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    n_batches = (n_records + batch_size - 1) // batch_size
    
    print(f"\n=== Generando {n_records} registros sinteticos para ML ===")
    print(f"    Cores: {n_jobs} | Batches: {n_batches} | Tamano batch: {batch_size}")
    if target_default_rate:
        print(f"    Tasa default objetivo: {target_default_rate*100:.1f}%")
    print("-" * 50)
    
    # Si hay datos base, preparar
    if not wealth_stage:
        wealth_cfg = None
    if base_df is not None and len(base_df) > 0:
        if len(base_df) < n_records:
            sampled = base_df.sample(n=n_records, replace=True, random_state=seed)
        else:
            sampled = base_df.sample(n=n_records, random_state=seed)
        if wealth_stage:
            wealth_cfg = wealth_cfg or wealth_profile.load_config()
            sampled = wealth_profile.resolve_inputs(
                sampled.reset_index(drop=True), wealth_cfg, None,
                np.random.default_rng(seed + STAGE2_SEED_OFFSET))
        all_rows = sampled.to_dict('records')
    else:
        if wealth_stage:
            print("[WARN] Sin datos base: stage 2 wealth omitido.")
        wealth_cfg = None
        all_rows = [{} for _ in range(n_records)]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_dir = OUTPUT_DIR / f"temp_{timestamp}"
    temp_dir.mkdir(exist_ok=True)
    
    flat_files = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_records)
        batch_rows = all_rows[start_idx:end_idx]
        
        # Semillas únicas por batch
        base_seed = random.randint(0, 1000000) + batch_idx * 100000
        args_list = [(row, base_seed + i, target_default_rate) for i, row in enumerate(batch_rows)]
        
        # Progreso en tiempo real
        print(f"  Batch {batch_idx + 1}/{n_batches} ({end_idx - start_idx} registros)...", end=" ", flush=True)
        
        if n_jobs == 1:
            flat_records = [_generate_flat_worker(a) for a in args_list]
        else:
            with Pool(n_jobs) as pool:
                flat_records = list(pool.imap(_generate_flat_worker, args_list, chunksize=100))
        
        flat_path = temp_dir / f"batch_{batch_idx:04d}.parquet"
        pd.DataFrame(flat_records).to_parquet(flat_path, index=False)
        flat_files.append(flat_path)
        del flat_records
        
        print(f"[OK] [{len(flat_files)} archivos]", flush=True)
    
    print("-" * 50)
    print("Combinando archivos finales...")
    
    # Combinar Parquets
    print(f"  Uniendo {len(flat_files)} archivos Parquet...", end=" ", flush=True)
    dfs = [pd.read_parquet(pf) for pf in flat_files]
    df_final = pd.concat(dfs, ignore_index=True)
    del dfs

    # Nombre archivo: incluye cantidad y tasa
    rate_suffix = f"{int(target_default_rate*100)}pct" if target_default_rate else "auto"
    filename_base = f"synthetic_{n_records}_{rate_suffix}_{timestamp}"

    # Stage 2 + 3: features wealth, reglas de consistencia / idoneidad, cohorte edge cases
    # (prints are ASCII-only so piped output works on cp1252 consoles)
    if wealth_cfg is not None:
        print("  Stage 2+3 wealth (features, CR01-CR07, MiFID II, EC01-EC09)...", flush=True)
        df_final, wealth_report = run_stage2(df_final, seed=seed + STAGE2_SEED_OFFSET, cfg=wealth_cfg,
                                             edge_case_share=edge_case_share)
        df_final, wealth_report["credit_reconciliation"] = reconcile_credit_fields(
            df_final, target_default_rate, seed)
        rec = wealth_report["credit_reconciliation"]
        print(f"    credit fields reconciled on {rec['rows_rebalanced']} rows; "
              f"debt mismatches {rec['debt_mismatch_rows']}")
        report_path = OUTPUT_DIR / f"{filename_base}_wealth_report.json"
        report_path.write_text(json.dumps(wealth_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    violations raw {sum(wealth_report['violations_raw'].values())} -> enforced "
              f"{sum(wealth_report['violations_enforced'].values())} "
              f"(exempted {wealth_report['violations_exempted']})")
        edge = wealth_report.get("edge_cases")
        if edge:
            per_id = " ".join(f"{k}={v['rows']}" for k, v in edge["profiles"].items())
            print(f"    edge cases {edge['achieved_total_share_pct']:.2f}% (target "
                  f"{edge['target_total_share_pct']:.2f}%): {per_id}")
        print(f"    report: {report_path.name}")

    final_parquet = OUTPUT_DIR / f"{filename_base}.parquet"
    df_final.to_parquet(final_parquet, index=False)
    print(f"[OK] {final_parquet.name}")
    
    final_csv = OUTPUT_DIR / f"{filename_base}.csv"
    df_final.to_csv(final_csv, index=False)
    print(f"  CSV: [OK] {final_csv.name}")
    
    # Limpiar temporales
    print("  Limpiando archivos temporales...", end=" ", flush=True)
    for f in temp_dir.glob("*"):
        f.unlink()
    temp_dir.rmdir()
    print("[OK]")
    
    return df_final


def menu():
    """Menú interactivo."""
    print("\n" + "=" * 60)
    print("GENERADOR DE DATOS BANCARIOS SINTETICOS - ESPANA")
    print("Dataset para ML con variable target (default)")
    print("=" * 60)
    print(f"CPUs disponibles: {cpu_count()}")
    
    # Cargar datos base
    base_df = load_base_data()
    
    # Cantidad de registros
    while True:
        try:
            n_input = input("\nCuantos registros generar? [100-1000000, default=1000]: ").strip()
            if n_input == '':
                n_records = 1000
            else:
                n_records = int(n_input)
            if 1 <= n_records <= 1000000:
                break
            print("Introduce un numero entre 1 y 1000000.")
        except ValueError:
            print("Introduce un numero valido.")
        except KeyboardInterrupt:
            print("\nHasta luego!")
            return
    
    # Tasa de default objetivo (permite múltiples)
    print("\nTasa de default objetivo?")
    print("  - 'auto': basada en score de riesgo (~6%)")
    print("  - numero: tasa fija (ej: 5 para 5%)")
    print("  - multiples: separadas por coma (ej: auto,5,10,15)")
    while True:
        try:
            rate_input = input("Tasa default [2-35, 'auto', o lista, default=auto]: ").strip().lower()
            if rate_input == '':
                target_rates = [None]  # auto
                break
            
            # Parsear múltiples tasas
            parts = [p.strip() for p in rate_input.split(',')]
            target_rates = []
            valid = True
            for p in parts:
                if p == 'auto':
                    target_rates.append(None)
                else:
                    rate = float(p)
                    if 2 <= rate <= 35:
                        target_rates.append(rate / 100)
                    else:
                        print(f"  '{p}' fuera de rango [2-35]")
                        valid = False
                        break
            if valid:
                break
        except ValueError:
            print("Formato invalido. Usa: auto, 5, o auto,5,10,15")
        except KeyboardInterrupt:
            print("\nHasta luego!")
            return
    
    # Número de cores
    max_cores = cpu_count()
    default_cores = max(1, max_cores - 1)
    while True:
        try:
            cores_input = input(f"\nCuantos cores usar? [1-{max_cores}, default={default_cores}]: ").strip()
            if cores_input == '':
                n_jobs = default_cores
            else:
                n_jobs = int(cores_input)
            if 1 <= n_jobs <= max_cores:
                break
            print(f"Introduce un numero entre 1 y {max_cores}.")
        except ValueError:
            print("Introduce un numero valido.")
        except KeyboardInterrupt:
            print("\nHasta luego!")
            return
    
    # Tamaño de batch (control de RAM)
    default_batch = 5000
    while True:
        try:
            batch_input = input(f"\nTamano de batch? (menor = menos RAM) [1000-50000, default={default_batch}]: ").strip()
            if batch_input == '':
                batch_size = default_batch
            else:
                batch_size = int(batch_input)
            if 100 <= batch_size <= 50000:
                break
            print("Introduce un numero entre 100 y 50000.")
        except ValueError:
            print("Introduce un numero valido.")
        except KeyboardInterrupt:
            print("\nHasta luego!")
            return
    
    # Generar para cada tasa
    n_batches = (n_records + batch_size - 1) // batch_size
    rates_str = ', '.join([f"{r*100:.0f}%" if r else "auto" for r in target_rates])
    print(f"\nGenerando {n_records} registros  {len(target_rates)} tasas = {n_records * len(target_rates)} total")
    print(f"Cores: {n_jobs} | Batch: {batch_size} | Tasas: [{rates_str}]")
    
    import time
    start_total = time.time()
    results = []
    
    for i, target_rate in enumerate(target_rates):
        rate_label = f"{target_rate*100:.0f}%" if target_rate else "auto(~6%)"
        print(f"\n{'='*60}")
        print(f"LOTE {i+1}/{len(target_rates)}: tasa={rate_label}")
        print(f"{'='*60}")
        
        start = time.time()
        result = generate_synthetic_data(n_records, base_df, n_jobs, batch_size, target_rate)
        elapsed = time.time() - start
        results.append(result)
        
        # Resumen por lote
        if isinstance(result, pd.DataFrame):
            n_def = result['default'].sum()
            tasa_real = n_def / len(result) * 100
            print(f"  -> Tasa real: {tasa_real:.2f}% | Tiempo: {elapsed:.1f}s")
    
    elapsed_total = time.time() - start_total
    
    print("\n" + "=" * 60)
    print("GENERACION COMPLETADA!")
    print(f"Archivos: {len(target_rates)} | Total: {elapsed_total:.1f}s")
    print(f"Guardados en: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Resumen final
    print("\nResumen de lotes generados:")
    for i, (rate, res) in enumerate(zip(target_rates, results)):
        if isinstance(res, pd.DataFrame):
            n_def = res['default'].sum()
            tasa_real = n_def / len(res) * 100
            rate_label = f"{rate*100:.0f}%" if rate else "auto"
            print(f"  [{i+1}] {n_records} regs | tasa_obj={rate_label} | tasa_real={tasa_real:.2f}%")


def main():
    """CLI entry point (risksynth-generate / python -m risksynth.generators.generate_synthetic)."""
    # Never crash on consoles whose code page lacks a character (e.g. cp1252 when piped)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    menu()


if __name__ == '__main__':
    main()
