"""
qqq_bot_completo.py
======================
Versión consolidada en UN SOLO ARCHIVO de todo el proyecto QQQ. Combina:

  1) BOT BÁSICO       - tendencia actual (SMA20 vs SMA50, velas diarias)
  2) BOT PREDICTIVO    - modelo de Machine Learning (Random Forest) que
                          predice la tendencia a 3 días con % de confianza
  3) ALERTA TEMPRANA    - monitoreo intradía (velas de 10-15 min) que avisa
                          con ~30 min de anticipación cuando el momentum de
                          corto plazo está por cambiar de signo

Cada sección funciona de forma independiente y se puede activar/desactivar
con las banderas ACTIVAR_* de la configuración. Todas comparten el mismo
bucle principal, cada una revisando en su propio intervalo.

NO ES ASESORÍA FINANCIERA. Ningún modelo aquí garantiza acierto: son
herramientas de apoyo, no promesas de rentabilidad.

REQUISITOS
----------
pip install yfinance requests scikit-learn pandas numpy joblib --break-system-packages

USO
---
python3 qqq_bot_completo.py

Si ACTIVAR_BOT_PREDICTIVO=True y no existe el archivo MODELO_ARCHIVO, el
script entrena el modelo automáticamente la primera vez que corre (tarda
uno o dos minutos). Para reentrenar manualmente, borra ese archivo .pkl.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
TICKER = "QQQ"
TELEGRAM_TOKEN = "8863282563:AAFlUYgh3hCSApSjQbU4fwGB-49GuaazX68"
TELEGRAM_CHAT_ID = "1297569950"

SOLO_HORARIO_MERCADO = True   # aplica a las 3 secciones
LOOP_SEGUNDOS = 60             # cada cuánto "late" el bucle principal (chequeo base)

# --- Activar/desactivar cada sección ---
ACTIVAR_BOT_BASICO = True
ACTIVAR_BOT_PREDICTIVO = False
ACTIVAR_ALERTA_TEMPRANA = True

# --- Sección 1: bot básico (SMA20/50, diario) ---
INTERVALO_BASICO_SEGUNDOS = 5 * 60
SOLO_SI_CAMBIA_BASICO = True

# --- Sección 2: modelo predictivo ML (diario) ---
INTERVALO_PREDICTIVO_SEGUNDOS = 30 * 60
SOLO_SI_CAMBIA_PREDICTIVO = True
CONFIANZA_MINIMA_PREDICTIVO = 0.0     # 0.0-1.0, sube para filtrar predicciones dudosas
MODELO_ARCHIVO = "modelo_qqq.pkl"
PERIODO_HISTORICO_ENTRENAMIENTO = "1y"
HORIZONTE_DIAS = 3
UMBRAL_MOVIMIENTO = 0.005

# --- Sección 3: alerta temprana intradía (10-15 min) ---
INTERVALO_ALERTA_SEGUNDOS = 5 * 60
VENTANA_HORAS_ALERTA = 24
MINUTOS_VELA = 15               # múltiplo de 5: 10, 15, 20...
EMA_RAPIDA = 3
EMA_LENTA = 8
VELAS_PENDIENTE = 4
MINUTOS_ANTICIPACION = 30
MINUTOS_ANTICIPACION_MIN = 5
COOLDOWN_MINUTOS_ALERTA = 30

FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d",
    "sma20_ratio", "sma50_ratio", "sma20_50_ratio",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "volat_10d",
    "vol_change",
]
CLASES = {0: "BAJISTA", 1: "LATERAL", 2: "ALCISTA"}


# ============================================================
# UTILIDADES COMUNES
# ============================================================
def en_horario_mercado() -> bool:
    ahora_ny = datetime.now(ZoneInfo("America/New_York"))
    if ahora_ny.weekday() >= 5:
        return False
    apertura = ahora_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    cierre = ahora_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return apertura <= ahora_ny <= cierre


def enviar_telegram(mensaje: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": mensaje, "parse_mode": "Markdown"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ============================================================
# SECCIÓN 1 - BOT BÁSICO (SMA20 vs SMA50, diario)
# ============================================================
def obtener_datos_diarios(ticker: str, periodo: str = "6mo") -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    t = yf.Ticker(ticker, session=session)
    data = t.history(period=periodo)
    if data.empty:
        raise RuntimeError(f"No se pudieron obtener datos para {ticker}")
    return data


def calcular_tendencia_sma(data: pd.DataFrame):
    d = data.copy()
    d["SMA20"] = d["Close"].rolling(window=20).mean()
    d["SMA50"] = d["Close"].rolling(window=50).mean()

    ultimo = d.iloc[-1]
    precio, sma20, sma50 = ultimo["Close"], ultimo["SMA20"], ultimo["SMA50"]

    if precio > sma20 > sma50:
        clave, emoji, texto = "ALCISTA", "📈", "Tendencia ALCISTA"
    elif precio < sma20 < sma50:
        clave, emoji, texto = "BAJISTA", "📉", "Tendencia BAJISTA"
    else:
        clave, emoji, texto = "LATERAL", "↔️", "Tendencia LATERAL / MIXTA"

    anterior = d.iloc[-2]["Close"]
    variacion_pct = (precio - anterior) / anterior * 100
    detalles = (
        f"Precio actual: ${precio:.2f} ({variacion_pct:+.2f}% hoy)\n"
        f"SMA20: ${sma20:.2f}\nSMA50: ${sma50:.2f}"
    )
    return clave, emoji, texto, detalles


def construir_mensaje_basico(emoji, texto, detalles):
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return (
        f"*{TICKER} - Reporte de tendencia (básico)*\n_{fecha}_\n\n"
        f"{emoji} *{texto}*\n\n{detalles}"
    )


def revisar_bot_basico(estado: dict):
    data = obtener_datos_diarios(TICKER)
    clave, emoji, texto, detalles = calcular_tendencia_sma(data)
    debe_enviar = (not SOLO_SI_CAMBIA_BASICO) or (clave != estado.get("ultima_clave"))
    if debe_enviar:
        enviar_telegram(construir_mensaje_basico(emoji, texto, detalles))
        estado["ultima_clave"] = clave
        print(f"[{datetime.now()}] [BÁSICO] Enviado: {texto}")
    else:
        print(f"[{datetime.now()}] [BÁSICO] Sin cambios ({texto}), no se envía.")


# ============================================================
# SECCIÓN 2 - MODELO PREDICTIVO ML (Random Forest, diario)
# ============================================================
def _rsi(precios: pd.Series, ventana: int = 14) -> pd.Series:
    delta = precios.diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    avg_gain = ganancia.ewm(alpha=1 / ventana, min_periods=ventana, adjust=False).mean()
    avg_loss = perdida.ewm(alpha=1 / ventana, min_periods=ventana, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def calcular_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    close = df["Close"]

    df["ret_1d"] = close.pct_change(1)
    df["ret_3d"] = close.pct_change(3)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    df["sma20_ratio"] = close / sma20 - 1
    df["sma50_ratio"] = close / sma50 - 1
    df["sma20_50_ratio"] = sma20 / sma50 - 1

    df["rsi14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    df["volat_10d"] = close.pct_change().rolling(10).std()

    if "Volume" in df.columns:
        vol_sma20 = df["Volume"].rolling(20).mean()
        df["vol_change"] = df["Volume"] / vol_sma20.replace(0, np.nan) - 1
    else:
        df["vol_change"] = 0.0

    return df


def crear_target(df: pd.DataFrame, horizonte: int, umbral: float) -> pd.Series:
    close = df["Close"]
    ret_futuro = close.shift(-horizonte) / close - 1
    target = pd.Series(1, index=df.index)  # LATERAL por defecto
    target[ret_futuro > umbral] = 2   # ALCISTA
    target[ret_futuro < -umbral] = 0  # BAJISTA
    target[ret_futuro.isna()] = np.nan
    return target


def entrenar_modelo():
    print(f"[ML] Descargando histórico ({PERIODO_HISTORICO_ENTRENAMIENTO}) y entrenando modelo...")
    data = obtener_datos_diarios(TICKER, PERIODO_HISTORICO_ENTRENAMIENTO)
    df = calcular_features(data)
    df["target"] = crear_target(df, HORIZONTE_DIAS, UMBRAL_MOVIMIENTO)
    df_limpio = df.dropna(subset=FEATURE_COLS + ["target"])

    X = df_limpio[FEATURE_COLS]
    y = df_limpio["target"].astype(int)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    modelo.fit(X_train_s, y_train)

    pred_test = modelo.predict(X_test_s)
    print(f"[ML] Accuracy en test: {accuracy_score(y_test, pred_test):.3f}")
    print(classification_report(
        y_test, pred_test,
        target_names=[CLASES[i] for i in sorted(CLASES)], zero_division=0,
    ))

    paquete = {
        "modelo": modelo, "scaler": scaler,
        "horizonte_dias": HORIZONTE_DIAS, "umbral": UMBRAL_MOVIMIENTO,
        "feature_cols": FEATURE_COLS,
    }
    joblib.dump(paquete, MODELO_ARCHIVO)
    print(f"[ML] Modelo guardado en {MODELO_ARCHIVO}")
    return paquete


def cargar_o_entrenar_modelo():
    if os.path.exists(MODELO_ARCHIVO):
        return joblib.load(MODELO_ARCHIVO)
    return entrenar_modelo()


def predecir_ml(paquete: dict):
    modelo, scaler, feature_cols = paquete["modelo"], paquete["scaler"], paquete["feature_cols"]
    data = obtener_datos_diarios(TICKER, "6mo")
    df = calcular_features(data)
    ultima_fila = df.dropna(subset=feature_cols).iloc[[-1]]

    X_s = scaler.transform(ultima_fila[feature_cols])
    pred = int(modelo.predict(X_s)[0])
    proba = {int(c): p for c, p in zip(modelo.classes_, modelo.predict_proba(X_s)[0])}
    precio_actual = float(ultima_fila["Close"].iloc[0])
    return pred, proba, precio_actual


def construir_mensaje_predictivo(pred, proba, precio, horizonte):
    emojis = {0: "📉", 1: "↔️", 2: "📈"}
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    confianza = proba[pred] * 100
    lineas_proba = "\n".join(f"  {CLASES[c]}: {p*100:.1f}%" for c, p in sorted(proba.items()))
    return (
        f"*{TICKER} - Predicción ML ({horizonte} days)*\n_{fecha}_\n\n"
        f"{emojis[pred]} *{CLASES[pred]}* (confianza: {confianza:.1f}%)\n\n"
        f"Precio actual: ${precio:.2f}\n\nProbabilidades:\n{lineas_proba}\n\n"
        f"_Modelo estadístico, no es asesoría financiera._"
    )


def revisar_bot_predictivo(paquete: dict, estado: dict):
    pred, proba, precio = predecir_ml(paquete)
    confianza = proba[pred]
    cumple_confianza = confianza >= CONFIANZA_MINIMA_PREDICTIVO
    debe_enviar = cumple_confianza and (
        (not SOLO_SI_CAMBIA_PREDICTIVO) or (pred != estado.get("ultima_pred"))
    )
    if debe_enviar:
        enviar_telegram(construir_mensaje_predictivo(pred, proba, precio, paquete["horizonte_dias"]))
        estado["ultima_pred"] = pred
        print(f"[{datetime.now()}] [PREDICTIVO] Enviado: {CLASES[pred]} ({confianza*100:.1f}%)")
    else:
        print(f"[{datetime.now()}] [PREDICTIVO] Sin envío ({CLASES[pred]}, {confianza*100:.1f}%).")


# ============================================================
# SECCIÓN 3 - ALERTA TEMPRANA INTRADÍA (velas de 10-15 min)
# ============================================================
def obtener_datos_intradia(ticker: str, horas: int, minutos_vela: int) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    t = yf.Ticker(ticker, session=session)
    base = t.history(period="5d", interval="5m")
    if base.empty:
        raise RuntimeError(f"No se pudieron obtener datos intradía para {ticker}")
    corte = base.index.max() - timedelta(hours=horas)
    base = base[base.index >= corte]
    agregado = base.resample(f"{minutos_vela}min", label="right", closed="right").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()
    return agregado


def calcular_indicadores_intradia(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema_rapida"] = d["Close"].ewm(span=EMA_RAPIDA, adjust=False).mean()
    d["ema_lenta"] = d["Close"].ewm(span=EMA_LENTA, adjust=False).mean()
    d["gap_pct"] = (d["ema_rapida"] - d["ema_lenta"]) / d["Close"] * 100
    return d


def detectar_alerta_temprana(df: pd.DataFrame, minutos_vela: int):
    ultimos = df.tail(VELAS_PENDIENTE + 1).copy()
    if len(ultimos) < VELAS_PENDIENTE + 1:
        return None

    deltas_min = ultimos.index.to_series().diff().dt.total_seconds() / 60
    if (deltas_min > minutos_vela * 3).any():
        return None

    gap_actual = ultimos["gap_pct"].iloc[-1]
    t0 = ultimos.index[0]
    x = (ultimos.index - t0).total_seconds().to_numpy() / 60.0
    y = ultimos["gap_pct"].to_numpy()
    pendiente, _ = np.polyfit(x, y, 1)

    tendencia_actual = "ALCISTA" if gap_actual > 0 else "BAJISTA"
    convergiendo = (gap_actual > 0 and pendiente < 0) or (gap_actual < 0 and pendiente > 0)
    if not convergiendo or pendiente == 0:
        return None

    minutos_para_cruce = -gap_actual / pendiente
    if not (MINUTOS_ANTICIPACION_MIN <= minutos_para_cruce <= MINUTOS_ANTICIPACION):
        return None

    tendencia_proyectada = "BAJISTA" if tendencia_actual == "ALCISTA" else "ALCISTA"
    return {
        "tendencia_actual": tendencia_actual,
        "tendencia_proyectada": tendencia_proyectada,
        "minutos_para_cruce": minutos_para_cruce,
        "gap_actual": gap_actual,
        "precio_actual": float(ultimos["Close"].iloc[-1]),
    }


def construir_mensaje_alerta(alerta: dict):
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return (
        f"⚠️ *{TICKER} - Posible cambio de tendencia*\n_{fecha}_ (velas de {MINUTOS_VELA} min)\n\n"
        f"Tendencia actual: {alerta['tendencia_actual']}\n"
        f"Proyección: podría virar a *{alerta['tendencia_proyectada']}* "
        f"en ~{alerta['minutos_para_cruce']:.0f} min\n\n"
        f"Precio actual: ${alerta['precio_actual']:.2f}\n"
        f"Gap EMA{EMA_RAPIDA}/EMA{EMA_LENTA}: {alerta['gap_actual']:+.3f}%\n\n"
        f"_Extrapolación de corto plazo, no una certeza. Verifica antes de actuar._"
    )


def revisar_alerta_temprana(estado: dict):
    data = obtener_datos_intradia(TICKER, VENTANA_HORAS_ALERTA, MINUTOS_VELA)
    df = calcular_indicadores_intradia(data)
    alerta = detectar_alerta_temprana(df, MINUTOS_VELA)

    if not alerta:
        print(f"[{datetime.now()}] [ALERTA] Sin señal de cruce inminente.")
        return

    ahora = datetime.now()
    ultima_ts = estado.get("ultima_alerta_ts")
    ultima_dir = estado.get("ultima_tendencia_proyectada")
    en_cooldown = (
        ultima_ts is not None
        and alerta["tendencia_proyectada"] == ultima_dir
        and (ahora - ultima_ts) < timedelta(minutes=COOLDOWN_MINUTOS_ALERTA)
    )
    if en_cooldown:
        print(f"[{ahora}] [ALERTA] Señal detectada pero en cooldown, no se reenvía.")
        return

    enviar_telegram(construir_mensaje_alerta(alerta))
    estado["ultima_alerta_ts"] = ahora
    estado["ultima_tendencia_proyectada"] = alerta["tendencia_proyectada"]
    print(f"[{ahora}] [ALERTA] Enviada: posible giro a {alerta['tendencia_proyectada']} "
          f"en {alerta['minutos_para_cruce']:.0f} min")


# ============================================================
# BUCLE PRINCIPAL
# ============================================================
def main():
    if MINUTOS_VELA % 5 != 0:
        print("⚠️  MINUTOS_VELA debe ser múltiplo de 5 (10, 15, 20...).")
        sys.exit(1)

    paquete_ml = None
    if ACTIVAR_BOT_PREDICTIVO:
        paquete_ml = cargar_o_entrenar_modelo()

    estado_basico, estado_predictivo, estado_alerta = {}, {}, {}
    proximo_basico = proximo_predictivo = proximo_alerta = datetime.now()

    activas = []
    if ACTIVAR_BOT_BASICO: activas.append("básico (SMA)")
    if ACTIVAR_BOT_PREDICTIVO: activas.append("predictivo (ML)")
    if ACTIVAR_ALERTA_TEMPRANA: activas.append(f"alerta temprana ({MINUTOS_VELA}min)")
    print(f"Bot completo iniciado para {TICKER}. Secciones activas: {', '.join(activas)}. "
          f"Ctrl+C para detener.")

    while True:
        try:
            mercado_ok = (not SOLO_HORARIO_MERCADO) or en_horario_mercado()
            ahora = datetime.now()

            if not mercado_ok:
                print(f"[{ahora}] Fuera de horario de mercado, saltando todas las secciones.")
            else:
                if ACTIVAR_BOT_BASICO and ahora >= proximo_basico:
                    revisar_bot_basico(estado_basico)
                    proximo_basico = ahora + timedelta(seconds=INTERVALO_BASICO_SEGUNDOS)

                if ACTIVAR_BOT_PREDICTIVO and ahora >= proximo_predictivo:
                    revisar_bot_predictivo(paquete_ml, estado_predictivo)
                    proximo_predictivo = ahora + timedelta(seconds=INTERVALO_PREDICTIVO_SEGUNDOS)

                if ACTIVAR_ALERTA_TEMPRANA and ahora >= proximo_alerta:
                    revisar_alerta_temprana(estado_alerta)
                    proximo_alerta = ahora + timedelta(seconds=INTERVALO_ALERTA_SEGUNDOS)

        except Exception as e:
            print(f"[{datetime.now()}] Error: {e}")

        time.sleep(LOOP_SEGUNDOS)


if __name__ == "__main__":
    main()
