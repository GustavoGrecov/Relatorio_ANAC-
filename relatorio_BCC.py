import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# Dicionário de coordenadas de aeródromos (use seu dicionário completo aqui)
COORDENADAS_MAP = {
    'SBAR': '-29.7106, -53.6892', 'SBGR': '-23.4356, -46.4731', 'SBMO': '-9.5108, -35.7917', 'SBGL': '-22.8100, -43.2506',
    'SAEZ': '-34.8222, -58.5358', 'SBRF': '-8.1264, -34.9228', 'SBSP': '-23.6261, -46.6564', 'SBSV': '-12.9108, -38.3311',
    'CYYZ': '43.6777, -79.6248', 'SBBR': '-15.8711, -47.9186', 'SBEG': '-3.0386, -60.0494', 'SBCT': '-25.5286, -49.1758',
    'SBFZ': '-3.7761, -38.5325', 'SBPA': '-29.9944, -51.1714', 'SBCF': '-19.6244, -43.9719', 'SBKG': '-7.2694, -35.8972',
    'SBFI': '-25.5961, -54.4856', 'SBKP': '-23.0069, -47.1344', 'SBAQ': '-21.8122, -48.1331', 'SDLO': '-21.6700, -49.8500',
    'SIRI': '-18.4289, -49.2217', 'SNCL': '-17.7289, -48.6111', 'SBVT': '-20.2581, -40.2858', 'LFPG': '49.0097, 2.5479',
    'SBJU': '-7.2206, -39.2736', 'SABE': '-34.5592, -58.4156', 'SBSG': '-5.7683, -35.3672', 'SBPS': '-16.4389, -39.0806',
    'SBFL': '-27.6705, -48.5525', 'SACO': '-31.3236, -64.2081', 'SBBE': '-1.3792, -48.4764', 'SBSL': '-2.5858, -44.2344',
    'SBTE': '-5.0594, -42.8236', 'LEMD': '40.4983, -3.5676', 'SBUR': '-18.8825, -48.2253', 'SBJE': '-2.9103, -40.3589',
    'SBPB': '-7.1483, -34.9506', 'SCEL': '-33.3930, -70.7858', 'SBUL': '-19.7650, -47.9650', 'SBGO': '-16.6322, -49.2217',
    'SEQM': '-0.1292, -78.3575', 'SKRG': '6.1644, -75.4231', 'SVMI': '10.6031, -66.9906', 'KMIA': '25.7959, -80.2870',
    'SBCB': '-22.9239, -42.0678', 'SBCG': '-20.4694, -54.6725', 'SBZM': '-21.5133, -43.1978', 'SBCY': '-15.6531, -56.1169',
    'SBRP': '-21.1342, -47.7742', 'SBRJ': '-22.9105, -43.1631', 'SBDN': '-16.6792, -39.0753', 'SBNF': '-26.8797, -48.6517',
    'SBPJ': '-10.2917, -48.3583', 'SAZS': '-38.7250, -62.1669', 'SBSR': '-20.8164, -49.4069', 'SBTT': '-4.8889, -52.3500',
    'SBIL': '-14.8156, -39.0333', 'SBCN': '-17.7289, -48.6111', 'SBAE': '-22.2483, -49.0567', 'SBJV': '-23.1811, -46.8847',
    'SBMA': '-12.6097, -41.6736', 'SOCA': '9.0792, -72.4347', 'SBSN': '-2.8839, -55.0050', 'LFPO': '48.7233, 2.3794',
    'SBCA': '-22.7322, -45.4622', 'KFLL': '26.0726, -80.1528', 'SBJP': '-7.1483, -34.9506', 'SWEI': '5.8708, -57.0514',
    'SBUA': '-22.8292, -53.2733', 'SBMK': '-14.2561, -42.2278', 'SULS': '-34.7842, -56.0275', 'LPPT': '38.7813, -9.1359',
    'MPTO': '8.9731, -79.3839', 'SBPL': '-6.6861, -38.2028', 'SBPV': '-8.7133, -63.9028', 'SBAU': '-21.2403, -50.4233',
    'SBMQ': '-2.4417, -44.2836', 'KJFK': '40.6413, -73.7781', 'SBMG': '-27.2247, -48.6072', 'SBVG': '-18.0892, -43.2508',
    'SAAR': '-32.9034, -60.7850', 'MMMX': '19.4361, -99.0720', 'SBCH': '-28.4778, -49.0069', 'KATL': '33.6407, -84.4277',
    'LIMC': '45.6300, 8.7233', 'SBLO': '-25.3478, -49.1834', 'SGAS': '-25.2556, -57.5194', 'SUMU': '-34.8339, -54.9194',
    'KIAD': '38.9469, -77.4559', 'SBTG': '-16.2347, -40.0056', 'SBPO': '-29.9944, -51.1714', 'SBGV': '-20.6536, -43.7681',
    'SBAT': '-22.7653, -43.8611', 'SBTF': '-16.7133, -39.0975', 'KORD': '41.9742, -87.9073', 'SBCZ': '-7.6000, -72.7667',
    'KEWR': '40.6925, -74.1687', 'LPPR': '41.2478, -8.6811', 'SBRB': '-9.9950, -67.8989', 'SBSJ': '-23.1792, -45.8717',
    'KPHX': '33.4339, -112.0117', 'MKJP': '17.9361, -76.7778', 'SVPR': '-9.7500, -67.5000', 'OMDB': '25.2528, 55.3644',
    'DGAA': '5.6047, -0.1650', 'OMDW': '24.8903, 55.3661', 'KIAH': '29.9844, -95.3414', 'GVAC': '16.8344, -22.9544',
    'GVPR': '14.9244, -23.4897', 'SARE': '-27.4861, -58.7647', 'LTFM': '41.2597, 28.7425', 'GOBD': '14.7392, -17.4900',
    'LSZH': '47.4583, 8.5492', 'EGLL': '51.4711, -0.4569', 'KLAX': '33.9425, -118.4081', 'DNMM': '6.5772, 3.3211',
    'SGES': '-20.4472, -50.2072', 'SBFN': '-6.3444, -47.4611', 'OTHH': '25.2731, 51.6081', 'SAWC': '-53.7778, -67.7500',
    'KMZJ': '33.3314, -111.6553', 'KTUS': '32.1161, -110.9411', 'SYCJ': '6.4978, -58.2542', 'SPJC': '-12.0219, -77.1144',
    'SMJP': '5.7528, -55.1875', 'SDSC': '-22.0056, -47.9033', 'SKBO': '4.7014, -74.1469', 'LEBL': '41.2969, 2.0783',
    'LIRF': '41.8003, 12.2389', 'SBJA': '-22.5306, -44.2944', 'SBBV': '2.8464, -60.6922', 'KMCO': '28.4312, -81.3090',
    'SBSM': '-29.6842, -53.7889', 'SBPK': '-29.1458, -56.0497', 'SBIP': '-19.1153, -47.9639', 'SBIZ': '-2.5181, -44.3306',
    'KBOS': '42.3642, -71.0053', 'SAME': '-31.6681, -68.4195', 'SBSI': '-10.9083, -37.0722', 'SBTC': '-23.4778, -51.1806',
    'SBPF': '-28.2417, -52.4292', 'EDDF': '50.0379, 8.5622', 'KVCV': '34.5964, -117.3833', 'EHAM': '52.3105, 4.7683',
    'ELLX': '49.6233, 6.2044', 'GMMN': '33.3672, -7.5889', 'SPSO': '-9.8567, -56.0706', 'FAOR': '-26.1392, 28.2461',
    'FACT': '-33.9694, -18.5972', 'EBBR': '50.9014, 4.4842', 'KSLC': '40.7884, -111.9781', 'SBCX': '-29.1958, -51.1883',
    'SLVR': '-17.7994, -63.1681', 'MMSM': '-21.0458, -104.7417', 'KRIV': '33.7319, -117.4556', 'FNLU': '-8.8583, 13.2311',
    'KAFW': '32.9792, -97.3194', 'SBUY': '-31.3033, -55.5328', 'MDPC': '18.4694, -68.3625', 'SWCA': '-9.2047, -61.8284',
    'PANC': '61.1744, -149.9961', 'RKSI': '37.4578, 126.4406', 'TNCA': '12.5014, -68.2694', 'MROC': '9.9939, -84.2089',
    'KHSV': '34.6364, -86.7750', 'MMUN': '21.0367, -86.8769', 'SBVC': '-20.9389, -40.3361', 'SBNM': '-10.3750, -67.8097',
    'KBFI': '47.5300, -122.3019', 'TFFF': '14.5922, -61.0022', 'KAEX': '31.2958, -92.5497', 'KMEM': '35.0425, -89.9767',
    'TJBQ': '18.3097, -65.6431', 'SBCJ': '-25.9389, -50.5542', 'SLCB': '-17.4244, -66.1514', 'SBDB': '-16.5417, -57.7083',
    'ZBAA': '40.0800, 116.5844', 'SBBH': '-19.8517, -43.9506', 'SBPP': '-22.1750, -51.4244', 'TNCC': '12.1886, -68.9597',
    'SNRU': '-27.8375, -50.3236', 'SNBR': '-12.1528, -45.0083', 'SWYN': '-3.2319, -61.7556', 'SBMY': '-7.1583, -48.2403',
    'SBAX': '-22.6011, -47.0514', 'SBMS': '-14.9195, -57.4556', 'SBJI': '-21.7889, -43.3653', 'SSKW': '-24.6806, -53.5028',
    'SBVH': '-7.9681, -59.8695', 'SBHT': '-3.8306, -45.4222', 'SNGI': '-18.5778, -49.1278', 'SBIH': '-17.7806, -42.8278',
    'EBLG': '50.6375, 5.4436', 'SWLC': '-12.9111, -45.7862', 'EDDM': '48.3539, 11.7862', 'SNCP': '-17.2347, -46.8722',
    'SBRD': '-19.9639, -44.5389', 'HAAB': '8.9778, 38.7992', 'SBTB': '-1.1292, -47.3556', 'SBML': '-26.3250, -52.6958',
    'SBLE': '-24.0764, -46.3736', 'SANT': '-34.8722, -56.1681', 'SNPD': '-16.7583, -43.8486', 'MTCH': '-6.0642, -8.1367',
    'SBMI': '-26.7028, -51.8700', 'CYUL': '45.4706, -73.7408', 'SDOW': '-23.4319, -47.3306',
    'SWBE': '-10.9389, -62.7195', 'SNIG': '-16.3417, -42.2000', 'SNWS': '-14.7083, -40.1181', 'SSOU': '-29.9472, -50.2153',
    'SNDV': '-18.1681, -45.4611', 'SNSS': '-12.2472, -38.9778', 'SNBA': '-13.2667, -43.4111', 'SSRS': '-29.4195, -51.9444',
    'SBJR': '-23.2347, -45.7139', 'SKCL': '10.3522, -75.1692', 'SNTO': '-10.7222, -37.2778', 'SNZR': '-11.2569, -40.3472',
    'SWTP': '-5.8278, -52.7917', 'SWKO': '-8.7806, -59.3556', 'SSUM': '-28.5806, -56.0472', 'SSGY': '-20.7681, -51.3306',
    'SSVL': '-27.6681, -50.3111', 'SSUV': '-27.8139, -49.6681', 'SNMZ': '-14.3806, -44.9528', 'SNVS': '-14.6583, -47.0472',
    'SBJD': '-23.4056, -46.5736', 'SDAG': '-22.2014, -45.4278', 'SBEK': '-16.3444, -54.7320', 'SBTU': '-2.7681, -49.6695',
    'SWLB': '-9.9695, -67.8222', 'SWPI': '-7.3195, -70.0139', 'SWBR': '-11.6347, -61.4486', 'SBMD': '-19.5862, -54.4700',
    'SNSM': '-13.3667, -38.9195', 'SWMW': '-8.1097, -62.7708', 'SNEB': '-6.5847, -43.2778', 'SBBW': '-12.0806, -45.9625',
    'KDFW': '32.8969, -97.0381', 'SBCP': '-11.1514, -40.5056', 'SNAB': '-12.9139, -41.2514', 'SNHS': '-12.6445, -39.0722',
    'SNGN': '-13.8306, -42.4472', 'SJZA': '-6.8750, -51.2250', 'SBPG': '-25.0862, -50.1472', 'SSGG': '-24.0306, -52.3056',
    'SIMK': '-11.2181, -49.1195', 'SNJM': '-12.5195, -41.8347', 'SASA': '-24.8556, -65.4862', 'SBSO': '-23.4778, -47.6167',
    'SBCR': '-19.5681, -57.1583', 'SBUG': '-29.7778, -57.0347', 'SNLN': '-13.5028, -39.0695', 'SWKQ': '1.2472, -61.8028',
    'MNMG': '12.1416, -86.1681', 'FLKK': '-15.3303, 28.4525', 'FLLS': '-15.4175, 28.3136', 'KTPA': '27.9754, -82.5333',
    'LBSF': '42.6950, 23.4061', 'LPLA': '38.7617, -27.0906', 'SARI': '-27.4497, -59.0561', 'LEZG': '41.6661, -1.0417',
    'GCLP': '27.9319, -15.3867', 'SNSV': '-19.2336, -43.9919',
}

mapa_nomes_aereas = {
    'GLO': 'GOL Linhas Aéreas', 'AZU': 'Azul Linhas Aéreas', 'TAM': 'LATAM Brasil',
    'PTB': 'VoePass', 'SID': 'Sideral Linhas Aéreas', 'TTL': 'Total Linhas Aéreas',
    'AJB': 'Abaeté Linhas Aéreas', 'MWM': 'Modern Logistics', 'ACA': 'Air Canada',
    'AAL': 'American Airlines', 'ARG': 'Aerolíneas Argentinas', 'AMX': 'Aeroméxico',
    'AVA': 'Avianca', 'BOV': 'Boliviana de Aviación', 'CMP': 'Copa Airlines',
    'DAL': 'Delta Air Lines', 'LAN': 'LATAM Chile', 'LAP': 'LATAM Paraguay',
    'LPE': 'LATAM Perú', 'ARE': 'LATAM Ecuador', 'LNE': 'LATAM Express',
    'SKU': 'Sky Airline', 'UAL': 'United Airlines', 'AFR': 'Air France',
    'AEA': 'Air Europa', 'BAW': 'British Airways', 'DLH': 'Lufthansa',
    'IBE': 'Iberia', 'ITY': 'ITA Airways', 'KLM': 'KLM Royal Dutch Airlines',
    'SWR': 'Swiss International Air Lines', 'TAP': 'TAP Air Portugal',
    'THY': 'Turkish Airlines', 'UAE': 'Emirates', 'QTR': 'Qatar Airways',
    'UPS': 'UPS Airlines', 'FDX': 'FedEx Express', 'GEC': 'Lufthansa Cargo',
    'TPA': 'Avianca Cargo',
}

def calcular_distancia(lat1, lon1, lat2, lon2):
    """Calcula a distância em km entre duas coordenadas geográficas."""
    R = 6371.0
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- FUNÇÃO PRINCIPAL DE CARREGAMENTO E LIMPEZA ---
@st.cache_data # Cache para alta performance: não recarrega e processa o arquivo a cada interação
def carregar_e_limpar_dados(filepath):
    """
    Carrega o arquivo CSV, corrige problemas de codificação, limpa e enriquece os dados.
    """
    # 1. Defina os nomes das colunas como deveriam ser, de forma limpa
    colunas_novas = [
        'icao_empresa_aerea', 'numero_voo', 'codigo_autorizacao_di', 'codigo_tipo_linha',
        'icao_aerodromo_origem', 'icao_aerodromo_destino', 'partida_prevista',
        'partida_real', 'chegada_prevista', 'chegada_real', 'situacao_voo',
        'codigo_justificativa'
    ]
    
    try:
        # 2. Carregue o arquivo CSV, aplicando os novos nomes e a codificação correta
        df = pd.read_csv(
            filepath, 
            sep=';', 
            encoding='latin1', ### SUPER IMPORTANTE PARA EVITAR ERROS DE FORMATAÇÃO - ME MATEI PARA DESCOBRIR ISSO 
            header=None,         # Não leia o cabeçalho do arquivo
            skiprows=2,          # Pule as duas primeiras linhas
            names=colunas_novas  # Aplique nossos nomes de coluna limpos
        )
    except FileNotFoundError:
        st.error(f"❌ ERRO: Arquivo '{filepath}' não encontrado.")
        return None

    # ETAPA 3: Processamento de coordenadas e cálculo de distância
    df['coords_origem'] = df['icao_aerodromo_origem'].astype(str).str.strip().map(COORDENADAS_MAP)
    df['coords_destino'] = df['icao_aerodromo_destino'].astype(str).str.strip().map(COORDENADAS_MAP)
    df.dropna(subset=['coords_origem', 'coords_destino'], inplace=True)
    
    df[['latitude_origem', 'longitude_origem']] = df['coords_origem'].str.split(', ', n=1, expand=True)
    df[['latitude_destino', 'longitude_destino']] = df['coords_destino'].str.split(', ', n=1, expand=True)
    
    geo_cols = ['latitude_origem', 'longitude_origem', 'latitude_destino', 'longitude_destino']
    for col in geo_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=geo_cols, inplace=True)
    
    df['distancia_km'] = calcular_distancia(
        df['latitude_origem'], df['longitude_origem'],
        df['latitude_destino'], df['longitude_destino']
    ).round(2)
    
    # ETAPA 4: Adicionar nomes completos das companhias
    df['nome_companhia'] = df['icao_empresa_aerea'].map(mapa_nomes_aereas).fillna(df['icao_empresa_aerea'])
    
    # ETAPA 5: Remover todas as colunas desnecessárias
    colunas_para_remover = [
        'icao_aerodromo_origem', 'icao_aerodromo_destino', 'icao_empresa_aerea',
        'codigo_justificativa', 'codigo_autorizacao_di', 'codigo_tipo_linha',
        'coords_origem', 'coords_destino'
    ]
    df_final = df.drop(columns=colunas_para_remover, errors='ignore')
    ### remover as colunas vazias
 

    return df_final
    

def setup_sidebar(df):
    """Configura a barra lateral com logo e filtros, usando os nomes completos das companhias."""
    try:
        image = Image.open("Ufabc_logo.png")
        st.sidebar.image(image, width=120)
    except FileNotFoundError:
        st.sidebar.warning("Arquivo 'Ufabc_logo.png' não encontrado.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📏 Filtrar por Distância (KM)")
    # Usa os nomes de coluna limpos
    min_dist = int(df['distancia_km'].min())
    max_dist = int(df['distancia_km'].max())
    distancia_selecionada = st.sidebar.slider(
        'Selecione um intervalo:', min_value=min_dist, max_value=max_dist, value=(min_dist, max_dist)
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ✈️ Filtrar por Companhia Aérea")
    
    # Usa a coluna 'nome_companhia' para uma experiência de usuário melhor
    todas_as_aereas = sorted(df['nome_companhia'].unique())
    aereas_nacionais = [
        'GOL Linhas Aéreas', 'Azul Linhas Aéreas', 'LATAM Brasil', 'VoePass',
        'Sideral Linhas Aéreas', 'Abaeté Linhas Aéreas', 'Total Linhas Aéreas'
    ]
    # Filtra para apenas as nacionais que de fato existem nos dados carregados
    aereas_nacionais_existentes = [nome for nome in aereas_nacionais if nome in todas_as_aereas]
    aereas_internacionais = [nome for nome in todas_as_aereas if nome not in aereas_nacionais_existentes]
    
    # Gerencia a seleção usando st.session_state
    if 'selecao_aerea' not in st.session_state:
        st.session_state.selecao_aerea = todas_as_aereas
        
    st.sidebar.write("Atalhos de seleção:")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("Apenas Nacionais", use_container_width=True):
        st.session_state.selecao_aerea = aereas_nacionais_existentes
    if col2.button("Apenas Internacionais", use_container_width=True):
        st.session_state.selecao_aerea = aereas_internacionais
        
    col3, col4 = st.sidebar.columns(2)
    if col3.button("Selecionar Todas", use_container_width=True):
        st.session_state.selecao_aerea = todas_as_aereas
    if col4.button("Limpar Seleção", use_container_width=True):
        st.session_state.selecao_aerea = []
        
    aereas_selecionadas = st.sidebar.multiselect(
        'Seleção manual:', options=todas_as_aereas, key='selecao_aerea'
    )
    return distancia_selecionada, aereas_selecionadas
#### ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------FUNÇÕES PRINCIPAIS PARA TAB DE VISÕES----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def gerar_grafico_atrasos_por_companhia(df):
    """
    Calcula e plota o atraso médio de partida e chegada para cada companhia aérea.

    A função considera apenas voos 'REALIZADO' e lida com dados ausentes.

    Args:
        df (pd.DataFrame): O DataFrame, que deve conter as colunas:
                           'situacao_voo', 'partida_prevista', 'partida_real',
                           'chegada_prevista', 'chegada_real' e 'nome_companhia'.

    Returns:
        matplotlib.figure.Figure: Um objeto de figura Matplotlib para ser usado
                                  com st.pyplot(), ou None se não houver dados.
    """
    # 1. Filtrar apenas voos realizados e garantir que não há dados nulos de tempo
    df_realizados = df[df['situacao_voo'] == 'REALIZADO'].copy()
    colunas_tempo = ['partida_prevista', 'partida_real', 'chegada_prevista', 'chegada_real']
    df_realizados.dropna(subset=colunas_tempo, inplace=True)

    # Se não houver voos válidos após a filtragem, retorna nada
    if df_realizados.empty:
        return None

    # 2. Garantir que as colunas são do tipo datetime (segurança extra)
    for col in colunas_tempo:
        df_realizados[col] = pd.to_datetime(df_realizados[col], errors='coerce')
    df_realizados.dropna(subset=colunas_tempo, inplace=True) # Remove novamente caso a conversão falhe

    # 3. Calcular os atrasos de partida e chegada em minutos
    df_realizados['atraso_partida_min'] = (df_realizados['partida_real'] - df_realizados['partida_prevista']).dt.total_seconds() / 60
    df_realizados['atraso_chegada_min'] = (df_realizados['chegada_real'] - df_realizados['chegada_prevista']).dt.total_seconds() / 60
    
    # 4. Agrupar por companhia e calcular a média dos atrasos
    media_atraso = df_realizados.groupby('nome_companhia')[['atraso_partida_min', 'atraso_chegada_min']].mean()
    
    # Ordenar pelo atraso na chegada para melhor visualização
    media_atraso = media_atraso.sort_values(by='atraso_chegada_min', ascending=False)

    # 5. Gerar o gráfico de barras comparativo
    fig, ax = plt.subplots(figsize=(14, 8)) # Tamanho generoso para boa visualização
    
    index = np.arange(len(media_atraso))
    bar_width = 0.35

    # Barras para Atraso na Partida
    ax.bar(index - bar_width/2, media_atraso['atraso_partida_min'], bar_width, 
           label='Atraso Médio na Partida', color='deepskyblue', alpha=0.8)
           
    # Barras para Atraso na Chegada
    ax.bar(index + bar_width/2, media_atraso['atraso_chegada_min'], bar_width, 
           label='Atraso Médio na Chegada', color='tomato', alpha=0.8)

    # 6. Configurar os detalhes do gráfico (títulos, rótulos, etc.)
    ax.set_title('Média de Atraso por Companhia Aérea (em Minutos)', fontsize=16)
    ax.set_xlabel('Companhia Aérea', fontsize=12)
    ax.set_ylabel('Atraso Médio (minutos)', fontsize=12)
    ax.set_xticks(index)
    ax.set_xticklabels(media_atraso.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Linha de referência no zero para ver voos adiantados vs. atrasados
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    fig.tight_layout()

    return fig
def gerar_grafico_cancelamentos(df_limpo):
    # 1. Filtra o DataFrame para manter apenas os voos com situação 'CANCELADO'
    df_cancelados = df_limpo[df_limpo['situacao_voo'] == 'CANCELADO']
    # 2. Usa groupby para contar os cancelamentos por companhia, ordena e pega o top 10
    top_10_data = (df_cancelados.groupby('nome_companhia')
                               .size()
                               .sort_values(ascending=False)
                               .head(10))

    # 3. Criar a figura e os eixos do gráfico
    fig, ax = plt.subplots(figsize=(12, 7))
    top_10_data.plot(kind='bar', color='salmon', ax=ax)

    # 4. Configurar os detalhes do gráfico
    ax.set_title('Top 10 Companhias com Mais Voos Cancelados', fontsize=16)
    ax.set_xlabel('Companhia Aérea', fontsize=12)
    ax.set_ylabel('Número de Voos Cancelados', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    return fig

def calcular_top_10_companhias(dataframe_limpo: pd.DataFrame):
    """
    Calcula os dados para o top 10 de companhias com mais voos.
    Esta função apenas prepara os dados, sem plotar.
    
    Args:
        dataframe_limpo (pd.DataFrame): O DataFrame já limpo e processado.
        
    Returns:
        pd.Series: Uma série com as 10 companhias com mais voos e suas contagens.
    """
    # 1. Contar e selecionar os dados
    contagem_por_companhia = dataframe_limpo['nome_companhia'].value_counts()
    top_10 = contagem_por_companhia.head(10)
    return top_10

def plotar_grafico_top_10(top_10_data: pd.Series):
    """
    Cria e retorna a figura de um gráfico de barras a partir dos dados do top 10.
    Esta função apenas plota, sem fazer cálculos.
    
    Args:
        top_10_data (pd.Series): Os dados já calculados do top 10.
        
    Returns:
        matplotlib.figure.Figure: O objeto da figura do gráfico para ser usado com st.pyplot().
    """
    # 1. Criar a figura e os eixos do gráfico
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 2. Plotar os dados no eixo 'ax'
    top_10_data.plot(kind='bar', ax=ax, color='darkcyan')

    # 3. Configurar os detalhes do gráfico
    ax.set_title('Top 10 Companhias com Mais Voos Registrados', fontsize=16)
    ax.set_xlabel('Companhia Aérea', fontsize=12)
    ax.set_ylabel('Número de Voos', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajusta o layout para garantir que nada seja cortado
    fig.tight_layout()

    # 4. Retorna o objeto da figura
    return fig


# --- FLUXO PRINCIPAL DO APLICATIVO ---

st.set_page_config(layout="wide")
st.title('Relatório Inteligente de Voos')

# 1. Carregamento e limpeza dos dados
df_limpo = carregar_e_limpar_dados('VRA_20251.csv')

# 2. Só continua se os dados foram carregados com sucesso
if df_limpo is not None:
    # 3. Configura a barra lateral e obtém as seleções do usuário
    distancia_selecionada, aereas_selecionadas = setup_sidebar(df_limpo)

    # 4. Aplica os filtros ao DataFrame para obter o resultado final
    df_filtrado_distancia = df_limpo[
        (df_limpo['distancia_km'] >= distancia_selecionada[0]) &
        (df_limpo['distancia_km'] <= distancia_selecionada[1])
    ]
    
    if aereas_selecionadas:
        df_filtrado_final = df_filtrado_distancia[df_filtrado_distancia['nome_companhia'].isin(aereas_selecionadas)]
    else:
        # Se nenhuma companhia for selecionada, mostra um DataFrame vazio
        df_filtrado_final = pd.DataFrame(columns=df_limpo.columns)

    # 5. Criação das Abas
    tab_mapas, tab_detalhes = st.tabs(["Visão Mapas", "Dados Detalhados"])

#### primeira ABA VISÃO MAPAS -------------------------------------------------------------------------------------------------------------#### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    with tab_mapas:
            ###contagem de voos 
        with st.container():
            st.markdown(f"#### Voos Encontrados: **{df_filtrado_final.shape[0]}**")
            st.markdown("---")
                
        with st.container():
            if not df_filtrado_final.empty:
                st.map(df_filtrado_final, latitude='latitude_origem', longitude='longitude_origem')
            else:
                st.info("Nenhum voo para exibir no mapa com os filtros atuais.")

#### SEGUNDA ABA VISÃO DETALHES -------------------------------------------------------------------------------------------------------------#### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
with tab_detalhes:
    # Container para a tabela de dados
    with st.container():
        st.markdown(f"#### Voos Encontrados: **{df_filtrado_final.shape[0]}**")
        st.markdown("---")
        st.markdown("### Tabela de Dados dos Voos Filtrados")
        
        # Mostra o DataFrame se não estiver vazio
        if not df_filtrado_final.empty:
            st.dataframe(df_filtrado_final, use_container_width=True)
        else:
            st.info("Nenhum dado encontrado para os filtros selecionados.")

    # Container para o gráfico de cancelamentos (alinhado com o container acima)
    with st.container():
        st.markdown("---")
        st.markdown("### Análise de Voos Cancelados (Top 10)")

        # Verifica se existem voos cancelados nos dados filtrados antes de gerar o gráfico
        if 'CANCELADO' in df_filtrado_final['situacao_voo'].unique():
            
            # Chama a sua função para gerar a figura do gráfico
            figura_cancelamentos = gerar_grafico_cancelamentos(df_filtrado_final)
            
            # Exibe o gráfico no Streamlit
            st.pyplot(figura_cancelamentos, use_container_width=True)
            
        else:
            st.info("Não há voos cancelados nos dados filtrados para exibir no gráfico.")


    with st.container(): ### grafico das comapnhias aereas top 10
        st.markdown('##### Top 10 Companhias Aéreas com mais voos')
            # Verifica se há dados para plotar
        if not df_filtrado_final.empty:
                dados_para_grafico = calcular_top_10_companhias(df_filtrado_final)
                figura_final = plotar_grafico_top_10(dados_para_grafico)
                st.pyplot(figura_final, use_container_width=True)
                st.markdown("---")
        else:
            st.info("Nenhum dado para exibir no gráfico com os filtros atuais.")
    # Exemplo de como adicionar em uma aba do seu app
    
    with st.container():
        st.markdown("### Análise de Atrasos por Companhia")
        
        # Chama a nova função com os dados filtrados
        figura_atrasos = gerar_grafico_atrasos_por_companhia(df_filtrado_final)
        
        # Verifica se o gráfico foi gerado antes de exibi-lo
        if figura_atrasos:
            st.pyplot(figura_atrasos, use_container_width=True)
        else:
            st.info("Não há dados de voos realizados para gerar o gráfico de atrasos com os filtros atuais.")

           