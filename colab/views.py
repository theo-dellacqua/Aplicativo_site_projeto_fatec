from django.contrib.auth.decorators import login_required
from django.shortcuts import render
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from django.db import connections
from colab.models import OEE_Prod_260521

@login_required
def grafico_1(request):
    produto_selecionado = 2

    queryset = (OEE_Prod_260521.objects.filter(produto=produto_selecionado).order_by('-inicio')[:5])

    dados = list(queryset.values('maquina','inicio','oee'))

    df = pd.DataFrame(dados)

    if df.empty:
        return render(request, "colab/grafico_1.html", {
            "grafico": None
        })

    df["inicio"] = pd.to_datetime(df["inicio"])

    df["OEE_MAQUINA"] = (df["maquina"].astype(str) + " - " + df["inicio"].dt.strftime('%d/%m %H:%M'))

    grafico_agg = (df.groupby("OEE_MAQUINA", as_index=False).agg(OEE_MAX=("oee", "mean")))

    grafico_agg = grafico_agg.sort_values("OEE_MAQUINA")

    plt.figure(figsize=(12, 8))

    barras = plt.bar(grafico_agg["OEE_MAQUINA"],grafico_agg["OEE_MAX"],color="#1f77b4")

    plt.title(f'OEE de execuções do produto por máquina - Produto {produto_selecionado}')
    plt.xlabel('Máquina / Execução')
    plt.ylabel('OEE')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0,float(grafico_agg["OEE_MAX"].max()) * 1.15)

    for barra in barras:
        altura = barra.get_height()

        plt.text(barra.get_x() + barra.get_width()/2,altura,f'{altura:.2f}',ha='center',va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.3)

    buffer = io.BytesIO()

    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()

    buffer.seek(0)

    grafico_png = base64.b64encode(buffer.getvalue()).decode()

    return render(request, "colab/grafico_1.html", {"grafico": grafico_png})



@login_required
def grafico_2(request):
    
    #Carregar dados
    query = """
        SELECT registro, produto, maquina, teep
        FROM colab_ega_kpis_prod
        WHERE produto = 2027 AND maquina = 23
        ORDER BY registro
    """

    df = pd.read_sql(query, connections['default'])
    df["registro"] = pd.to_datetime(df["registro"])

    #Calcular INTERVALO entre execuções
    df["intervalo"] = df["registro"].diff().dt.days
    intervalos = df["intervalo"].dropna()

    #Criar eixo X como números sequenciais
    x_int = np.arange(len(intervalos))
    y_int = intervalos.values

    #REGRASSÃO LINEAR SIMPLES para prever intervalo
    coef = np.polyfit(x_int, y_int, 1)
    a, b = coef

    proximo_intervalo = a * (len(intervalos)) + b

    #Calcular data prevista
    ultima_data = df["registro"].max()
    data_prevista = ultima_data + pd.Timedelta(days=float(proximo_intervalo))

    #Previsão do TEEP usando regressão linear simples
    serie_teep = df.set_index("registro")["teep"]

    x_teep = np.arange(len(serie_teep))
    y_teep = serie_teep.values

    coef_teep = np.polyfit(x_teep, y_teep, 1)
    a2, b2 = coef_teep

    teep_previsto = a2 * len(serie_teep) + b2

    #Criar gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(serie_teep.index, serie_teep, label="TEEP", color="green")

    # ponto previsto
    plt.scatter([data_prevista], [teep_previsto], color="red", s=120, label="Previsão do TEEP")

    # linha tracejada até previsão
    plt.plot(
        [serie_teep.index[-1], data_prevista],
        [serie_teep.iloc[-1], teep_previsto],
        linestyle="--",
        color="gray"
    )

    plt.title("Previsão do TEEP do Produto 2027 na Máquina 23 na próxima execução")
    plt.xlabel("Data")
    plt.ylabel("TEEP (%)")
    plt.grid(True)
    plt.legend()

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    grafico_png = base64.b64encode(buffer.getvalue()).decode()

    return render(request, "colab/grafico_2.html", {
        "grafico": grafico_png,
        "data_prevista": data_prevista,
        "teep_previsto": round(float(teep_previsto), 2),
        "ultima_data": ultima_data
    })


@login_required
def grafico_3(request):
    
    # Carregar dados
    query = """
        SELECT registro, produto, maquina, qualidade
        FROM colab_ega_kpis_prod
        WHERE produto = 2027 AND maquina = 23
        ORDER BY registro
    """

    df = pd.read_sql(query, connections['default'])
    df["registro"] = pd.to_datetime(df["registro"])

    # Calcular INTERVALO entre execuções
    df["intervalo"] = df["registro"].diff().dt.days
    intervalos = df["intervalo"].dropna()

    # Criar eixo X de números sequenciais
    x_int = np.arange(len(intervalos))
    y_int = intervalos.values

    # Regressão Linear p/ prever intervalo
    coef = np.polyfit(x_int, y_int, 1)
    a, b = coef

    proximo_intervalo = a * (len(intervalos)) + b

    # Calcular a data prevista
    ultima_data = df["registro"].max()
    data_prevista = ultima_data + pd.Timedelta(days=float(proximo_intervalo))

    # Previsão da QUALIDADE
    serie_qual = df.set_index("registro")["qualidade"]

    x_q = np.arange(len(serie_qual))
    y_q = serie_qual.values

    coef_q = np.polyfit(x_q, y_q, 1)
    a2, b2 = coef_q

    qualidade_prevista = a2 * len(serie_qual) + b2

    # Criar gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(serie_qual.index, serie_qual, label="Qualidade", color="orange")

    # ponto previsto
    plt.scatter([data_prevista], [qualidade_prevista], color="red", s=120, label="Previsão da Qualidade")

    # linha tracejada até previsão
    plt.plot(
        [serie_qual.index[-1], data_prevista],
        [serie_qual.iloc[-1], qualidade_prevista],
        linestyle="--",
        color="gray"
    )

    plt.title("Previsão da Qualidade do Produto 2027 na Máquina 23 na próxima execução")
    plt.xlabel("Data")
    plt.ylabel("Qualidade (%)")
    plt.grid(True)
    plt.legend()

    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    grafico_png = base64.b64encode(buffer.getvalue()).decode()

    return render(request, "colab/grafico_3.html", {
        "grafico": grafico_png,
        "data_prevista": data_prevista,
        "qualidade_prevista": round(float(qualidade_prevista), 2),
        "ultima_data": ultima_data
    })



@login_required
def powerbi(request):
    powerbi_link = "https://app.powerbi.com/view?r=eyJrIjoiODgyZmU5ZDEtNWQxYi00N2YwLTk5ZWMtZDMwYmU1ZWE2ZjVhIiwidCI6ImNmNzJlMmJkLTdhMmItNDc4My1iZGViLTM5ZDU3YjA3Zjc2ZiIsImMiOjR9"
    return render(request, "colab/powerbi.html", {
        "powerbi_url": powerbi_link
    })

