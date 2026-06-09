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
from colab.models import ITENS_260521
import matplotlib.dates as mdates


@login_required
def grafico_1(request):

    produtos = (OEE_Prod_260521.objects.values_list('produto', flat=True).distinct().order_by('produto'))

    produto_selecionado = request.GET.get("produto")

    if produto_selecionado:
        produto_selecionado = int(produto_selecionado)
    else:
        produto_selecionado = produtos.first()

    queryset = (OEE_Prod_260521.objects.filter(produto=produto_selecionado).order_by('-inicio')[:5])

    dados = list(queryset.values('maquina','inicio','oee'))

    df = pd.DataFrame(dados)

    if df.empty:
        return render(request,"colab/grafico_1.html",{"grafico": None,"produtos": produtos,"produto_selecionado": produto_selecionado})

    df["inicio"] = pd.to_datetime(df["inicio"])

    df["inicio"] = (df["inicio"].dt.tz_convert("America/Sao_Paulo"))

    df["OEE_MAQUINA"] = (df["maquina"].astype(str) + " - " + df["inicio"].dt.strftime('%d/%m %H:%M'))

    grafico_agg = (df.groupby("OEE_MAQUINA",as_index=False).agg(OEE_MAX=("oee", "mean")))

    grafico_agg = grafico_agg.sort_values("OEE_MAQUINA")

    plt.figure(figsize=(12, 8))

    barras = plt.bar(grafico_agg["OEE_MAQUINA"],grafico_agg["OEE_MAX"],color="#1f77b4")

    plt.title(f'OEE de execuções do produto por máquina - Produto {produto_selecionado}')
    plt.xlabel('Máquina / Execução')
    plt.ylabel('OEE')
    plt.xticks(rotation=45,ha='right')
    plt.ylim(0,float(grafico_agg["OEE_MAX"].max()) * 1.15)

    for barra in barras:

        altura = barra.get_height()

        plt.text(barra.get_x() + barra.get_width()/2,altura,f'{altura:.2f}',ha='center',va='bottom')

    plt.grid(axis='y',linestyle='--',alpha=0.3)

    buffer = io.BytesIO()

    plt.tight_layout()

    plt.savefig(buffer,format='png')
    plt.close()

    buffer.seek(0)

    grafico_png = base64.b64encode(buffer.getvalue()).decode()

    return render(request,"colab/grafico_1.html",{"grafico": grafico_png,"produtos": produtos,"produto_selecionado": produto_selecionado})



@login_required
def grafico_2(request):

    produtos = (OEE_Prod_260521.objects.values_list('produto', flat=True).distinct().order_by('produto'))

    produto_filtro = request.GET.get("produto")

    if produto_filtro:
        produto_filtro = int(produto_filtro)
    else:
        produto_filtro = produtos.first()

    queryset = (OEE_Prod_260521.objects.filter(produto=produto_filtro))

    dados = list(queryset.values('maquina','oee'))

    df = pd.DataFrame(dados)

    if df.empty:
        return render(request,"colab/grafico_2.html",{"grafico": None,"produtos": produtos,"produto_selecionado": produto_filtro})

    oee_maquina = (df.groupby("maquina")["oee"].mean().reset_index().sort_values(by="oee", ascending=False))

    plt.figure(figsize=(12, 6))

    sns.barplot(data=oee_maquina,x="maquina",y="oee",color="#2e8b57")

    plt.title(f"OEE Produto {produto_filtro} por Máquina",fontsize=16)
    plt.xlabel("Máquina")
    plt.ylabel("OEE (%)")
    plt.xticks(rotation=45)
    plt.ylim(0,float(oee_maquina["oee"].max()) + 10)

    for index, row in oee_maquina.iterrows():

        plt.text(index,float(row["oee"]) + 1,f'{float(row["oee"]):.1f}%',ha='center')

    buffer = io.BytesIO()

    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()

    buffer.seek(0)

    grafico_png = base64.b64encode(buffer.getvalue()).decode()

    return render(request,"colab/grafico_2.html",{"grafico": grafico_png,"produtos": produtos,"produto_selecionado": produto_filtro})





@login_required
def grafico_3(request):

    queryset = (ITENS_260521.objects.exclude(pecas_hora__isnull=True).exclude(inicio__isnull=True))

    dados = list(queryset.values('inicio', 'pecas_hora'))

    df = pd.DataFrame(dados)

    if df.empty:
        return render(request, "colab/grafico_3.html", {"grafico": None})

    df["inicio"] = pd.to_datetime(df["inicio"])
    df["pecas_hora"] = pd.to_numeric(df["pecas_hora"], errors="coerce")

    df = df.dropna(subset=["inicio", "pecas_hora"])

    media_pecas_hora = df["pecas_hora"].mean()

    data_inicio = request.GET.get("data_inicio", "2026-05-21T10:59")
    quantidade = int(request.GET.get("quantidade", 20000))

    inicio = pd.to_datetime(data_inicio)

    oee = 66.66

    tempo_ideal_horas = quantidade / media_pecas_hora

    tempo_real_horas = tempo_ideal_horas / (oee / 100)

    fim_previsto = inicio + pd.to_timedelta(tempo_real_horas,unit='h')

    fig, ax = plt.subplots(figsize=(15, 4))

    ax.plot([inicio, fim_previsto],[0, 0],linewidth=12)

    ax.scatter(inicio, 0, s=350)
    ax.scatter(fim_previsto, 0, s=350)

    ax.text(inicio,0.05,'INÍCIO\n' + inicio.strftime('%d/%m/%Y %H:%M'),fontsize=11)

    ax.text(fim_previsto,0.05,'PREVISÃO FINAL\n' + fim_previsto.strftime('%d/%m/%Y %H:%M'),fontsize=11,ha='right')

    meio = inicio + ((fim_previsto - inicio) / 2)

    texto = (
        f'Quantidade: {quantidade} peças\n'
        f'Média Peças/Hora: {media_pecas_hora:.2f}\n'
        f'OEE: {oee}%\n'
        f'Tempo Previsto: {tempo_real_horas:.2f} horas'
    )

    ax.text(meio,-0.05,texto,fontsize=12,ha='center',bbox=dict(boxstyle='round', pad=0.5))

    ax.set_yticks([])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))

    plt.xticks(rotation=20)
    plt.title('Previsão de Produção')
    plt.xlabel('Data')
    plt.grid(True)

    buffer = io.BytesIO()

    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close()

    buffer.seek(0)

    grafico_png = base64.b64encode(buffer.getvalue()).decode()

    return render(request,"colab/grafico_3.html",{"grafico": grafico_png,"media_pecas_hora": round(float(media_pecas_hora), 2),"tempo_ideal": round(float(tempo_ideal_horas), 2),"tempo_real": round(float(tempo_real_horas), 2),"fim_previsto": fim_previsto,"data_inicio": data_inicio,"quantidade": quantidade})



@login_required
def powerbi(request):
    powerbi_link = "https://app.powerbi.com/view?r=eyJrIjoiODgyZmU5ZDEtNWQxYi00N2YwLTk5ZWMtZDMwYmU1ZWE2ZjVhIiwidCI6ImNmNzJlMmJkLTdhMmItNDc4My1iZGViLTM5ZDU3YjA3Zjc2ZiIsImMiOjR9"
    return render(request, "colab/powerbi.html", {
        "powerbi_url": powerbi_link
    })

