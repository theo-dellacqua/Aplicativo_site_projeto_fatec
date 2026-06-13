from django.contrib.auth.decorators import login_required
from django.shortcuts import render
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from django.db import connections
from colab.models import OEE_Prod_260611
from colab.models import ITENS_260611
from colab.models import PRODUTOS_260610
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from datetime import timedelta

import matplotlib
matplotlib.use("Agg")



@login_required
def grafico_1(request):

    produtos = (OEE_Prod_260611.objects.values_list('produto', flat=True).distinct().order_by('produto'))

    produto_selecionado = request.GET.get("produto")

    if produto_selecionado:
        produto_selecionado = int(produto_selecionado)
    else:
        produto_selecionado = produtos.first()

    queryset = (OEE_Prod_260611.objects.filter(produto=produto_selecionado).order_by('-inicio')[:5])

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

    produtos = (OEE_Prod_260611.objects.values_list('produto', flat=True).distinct().order_by('produto'))

    produto_filtro = request.GET.get("produto")

    if produto_filtro:
        produto_filtro = int(produto_filtro)
    else:
        produto_filtro = produtos.first()

    queryset = (OEE_Prod_260611.objects.filter(produto=produto_filtro))

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







def prever_arima_simples(serie):

    serie = np.array(serie, dtype=float)

    if len(serie) < 3:
        return float(serie[-1])

    diff = np.diff(serie)

    x = diff[:-1]
    y = diff[1:]

    denominador = np.sum(x ** 2)

    if denominador == 0:
        phi = 0
    else:
        phi = np.sum(x * y) / denominador

    ultimo_diff = diff[-1]

    previsao_diff = phi * ultimo_diff

    previsao = serie[-1] + previsao_diff

    return float(previsao)



@login_required
def grafico_3(request):

    produtos = (
        PRODUTOS_260610.objects
        .exclude(peca__isnull=True)
        .values_list('peca', flat=True)
        .distinct()
        .order_by('peca')
    )

    produto = request.GET.get("produto")
    maquina = request.GET.get("maquina")
    operacao = request.GET.get("operacao")
    data_inicio = request.GET.get("data_inicio", "")
    quantidade = request.GET.get("quantidade", "")
    modelo = request.GET.get("modelo", "")

    if not produto:

        return render(
            request,
            "colab/grafico_3.html",
            {
                "grafico": None,
                "produtos": produtos,
                "maquinas": [],
                "operacoes": [],
                "produto": None,
                "maquina": None,
                "operacao": None,
                "modelo": modelo,
                "data_inicio": "",
                "quantidade": ""
            }
        )

    produto = int(produto)

    base_oee = OEE_Prod_260611.objects.filter(produto=produto)

    base_maquinas = base_oee

    if operacao:
        base_maquinas = base_maquinas.filter(operacao=int(operacao))

    maquinas = list(
        base_maquinas
        .exclude(maquina__isnull=True)
        .values_list("maquina", flat=True)
        .distinct()
        .order_by("maquina")
    )

    if maquina and int(maquina) not in maquinas:
        maquinas.append(int(maquina))
        maquinas.sort()

    base_operacoes = base_oee

    if maquina:
        base_operacoes = base_operacoes.filter(maquina=int(maquina))

    operacoes = list(
        base_operacoes
        .exclude(operacao__isnull=True)
        .values_list("operacao", flat=True)
        .distinct()
        .order_by("operacao")
    )

    if operacao and int(operacao) not in operacoes:
        operacoes.append(int(operacao))
        operacoes.sort()

    if not maquina or not operacao or not modelo:

        return render(
            request,
            "colab/grafico_3.html",
            {
                "grafico": None,
                "produtos": produtos,
                "maquinas": maquinas,
                "operacoes": operacoes,
                "produto": produto,
                "maquina": maquina,
                "operacao": operacao,
                "modelo": modelo,
                "data_inicio": data_inicio,
                "quantidade": quantidade
            }
        )

    maquina = int(maquina)
    operacao = int(operacao)

    if not data_inicio or not quantidade:

        return render(
            request,
            "colab/grafico_3.html",
            {
                "grafico": None,
                "produtos": produtos,
                "maquinas": maquinas,
                "operacoes": operacoes,
                "produto": produto,
                "maquina": maquina,
                "operacao": operacao,
                "modelo": modelo,
                "data_inicio": data_inicio,
                "quantidade": quantidade
            }
        )

    quantidade = int(quantidade)

    registro_produto = (
        PRODUTOS_260610.objects
        .filter(peca=produto, pecas_hora__isnull=False)
        .first()
    )

    if not registro_produto:

        return render(
            request,
            "colab/grafico_3.html",
            {
                "grafico": None,
                "produtos": produtos,
                "maquinas": maquinas,
                "produto": produto,
                "maquina": maquina,
                "modelo": modelo,
                "data_inicio": data_inicio,
                "quantidade": quantidade
            }
        )

    pecas_hora = float(registro_produto.pecas_hora)

    historico_oee = (
        OEE_Prod_260611.objects
        .filter(
            produto=produto,
            maquina=maquina,
            operacao=operacao,
            oee__isnull=False
        )
        .order_by('inicio')
    )

    df_oee = pd.DataFrame(list(historico_oee.values('oee')))

    if df_oee.empty:

        return render(
            request,
            "colab/grafico_3.html",
            {
                "grafico": None,
                "produtos": produtos,
                "maquinas": maquinas,
                "operacoes": operacoes,
                "produto": produto,
                "maquina": maquina,
                "operacao": operacao,
                "modelo": modelo,
                "data_inicio": data_inicio,
                "quantidade": quantidade,
                "mensagem": "Não existem dados de OEE para este produto e máquina."
            }
        )

    serie_oee = df_oee["oee"].astype(float)

    # =========================
    # MODELOS
    # =========================

    if len(serie_oee) == 1:
        oee_previsto = float(serie_oee.iloc[0])

    else:

        X = np.arange(len(serie_oee)).reshape(-1, 1)
        y = serie_oee.values.reshape(-1, 1)

        if modelo == "arima":

            oee_previsto = prever_arima_simples(serie_oee.values)

        elif modelo == "svr":

            scaler_x = StandardScaler()
            scaler_y = StandardScaler()

            X_scaled = scaler_x.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y).ravel()

            svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01)
            svr.fit(X_scaled, y_scaled)

            pred = svr.predict(scaler_x.transform([[len(serie_oee)]]))
            oee_previsto = scaler_y.inverse_transform([pred])[0][0]

        else:

            regressao = LinearRegression()
            regressao.fit(X, serie_oee)

            oee_previsto = regressao.predict([[len(serie_oee)]])[0]

    inicio = pd.to_datetime(data_inicio)

    tempo_ideal_horas = quantidade / pecas_hora

    tempo_real_horas = tempo_ideal_horas / (oee_previsto / 100)

    fim_previsto = inicio + pd.to_timedelta(tempo_real_horas, unit='h')

    fig, ax = plt.subplots(figsize=(15, 4))

    ax.plot([inicio, fim_previsto], [0, 0], linewidth=12)
    ax.scatter(inicio, 0, s=350)
    ax.scatter(fim_previsto, 0, s=350)

    ax.text(inicio, 0.05,
            'INÍCIO\n' + inicio.strftime('%d/%m/%Y %H:%M'),
            fontsize=11)

    ax.text(fim_previsto, 0.05,
            'PREVISÃO FINAL\n' + fim_previsto.strftime('%d/%m/%Y %H:%M'),
            fontsize=11, ha='right')

    meio = inicio + ((fim_previsto - inicio) / 2)

    texto = (
        f'Produto: {produto}\n'
        f'Máquina: {maquina}\n'
        f'Operação: {operacao}\n'
        f'Modelo: {modelo.upper()}\n'
        f'Quantidade: {quantidade} peças\n'
        f'Peças/Hora: {pecas_hora:.2f}\n'
        f'OEE Previsto: {oee_previsto:.2f}%\n'
        f'Tempo Previsto: {tempo_real_horas:.2f} horas'
    )

    ax.text(meio, -0.05, texto,
            fontsize=12, ha='center',
            bbox=dict(boxstyle='round', pad=0.5))

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

    return render(
        request,
        "colab/grafico_3.html",
        {
            "grafico": grafico_png,
            "produto": produto,
            "maquina": maquina,
            "operacao": operacao,
            "modelo": modelo,
            "produtos": produtos,
            "maquinas": maquinas,
            "operacoes": operacoes,
            "pecas_hora": round(pecas_hora, 2),
            "oee_previsto": round(oee_previsto, 2),
            "tempo_ideal": round(tempo_ideal_horas, 2),
            "tempo_real": round(tempo_real_horas, 2),
            "fim_previsto": fim_previsto,
            "data_inicio": data_inicio,
            "quantidade": quantidade
        }
    )



@login_required
def grafico_4(request):

    maquinas = list(
        ITENS_260611.objects
        .exclude(maquina__isnull=True)
        .values_list("maquina", flat=True)
        .distinct()
        .order_by("maquina")
    )

    maquina = request.GET.get("maquina")
    operacao = request.GET.get("operacao")
    produto = request.GET.get("produto")
    quantidade = request.GET.get("quantidade")
    modelo = request.GET.get("modelo", "")

    operacoes = []
    produtos = []

    if maquina:
        operacoes = list(
            ITENS_260611.objects
            .filter(maquina=int(maquina))
            .exclude(operacao__isnull=True)
            .exclude(operacao=0)
            .values_list("operacao", flat=True)
            .distinct()
            .order_by("operacao")
        )

    produtos = list(
        ITENS_260611.objects
        .exclude(peca__isnull=True)
        .values_list("peca", flat=True)
        .distinct()
        .order_by("peca")
    )

    # ❌ SEM MENSAGEM AQUI
    if not operacao or not produto or not quantidade:
        return render(request, "colab/grafico_4.html", {
            "grafico": None,
            "maquinas": maquinas,
            "operacoes": operacoes,
            "produtos": produtos,
            "maquina": maquina,
            "operacao": operacao,
            "produto": produto,
            "modelo": modelo,
            "quantidade": quantidade
        })

    maquina = int(maquina) if maquina else None
    operacao = int(operacao)
    produto = int(produto)
    quantidade = int(quantidade)

    fila_query = ITENS_260611.objects.filter(
        operacao=operacao
    ).exclude(operacao=0)

    if maquina:
        fila_query = fila_query.filter(maquina=maquina)

    fila = list(fila_query.order_by("sequencia"))

    if not fila:
        return render(request, "colab/grafico_4.html", {
            "grafico": None,
            "maquinas": maquinas,
            "operacoes": operacoes,
            "produtos": produtos,
            "maquina": maquina,
            "operacao": operacao,
            "produto": produto,
            "modelo": modelo,
            "quantidade": quantidade
        })

    historico_oee = OEE_Prod_260611.objects.filter(
        operacao=operacao,
        produto=produto,
        oee__isnull=False
    )

    if maquina:
        historico_oee = historico_oee.filter(maquina=maquina)

    historico_oee = historico_oee.order_by("inicio")

    df_oee = pd.DataFrame(list(historico_oee.values("oee")))

    if df_oee.empty:
        return render(request, "colab/grafico_4.html", {
            "grafico": None,
            "maquinas": maquinas,
            "operacoes": operacoes,
            "produtos": produtos,
            "maquina": maquina,
            "operacao": operacao,
            "produto": produto,
            "modelo": modelo,
            "quantidade": quantidade
        })

    serie_oee = df_oee["oee"].astype(float)

    if len(serie_oee) == 1:
        oee_previsto = float(serie_oee.iloc[0])
    else:
        X = np.arange(len(serie_oee)).reshape(-1, 1)
        y = serie_oee.values.reshape(-1, 1)

        if modelo == "arima":
            oee_previsto = prever_arima_simples(serie_oee.values)

        elif modelo == "svr":
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()

            Xs = scaler_x.fit_transform(X)
            ys = scaler_y.fit_transform(y).ravel()

            svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01)
            svr.fit(Xs, ys)

            pred = svr.predict(scaler_x.transform([[len(serie_oee)]]))
            oee_previsto = scaler_y.inverse_transform([pred])[0][0]

        else:
            model = LinearRegression()
            model.fit(X, serie_oee)
            oee_previsto = model.predict([[len(serie_oee)]])[0]

    oee_previsto = max(oee_previsto, 1)

    previsoes = []
    data_anterior = None

    for idx, item in enumerate(fila):

        quantidade_restante = max(
            0,
            (item.programado or 0) - (item.produzido or 0)
        )

        pecas_hora_item = float(item.pecas_hora or 1)

        tempo_ideal = quantidade_restante / pecas_hora_item
        tempo_real = tempo_ideal / (oee_previsto / 100)

        if idx == 0:
            inicio_prev = item.prev_inicio or item.inicio
        else:
            inicio_prev = data_anterior

        if not inicio_prev:
            continue

        fim_prev = inicio_prev + timedelta(hours=tempo_real)

        previsoes.append({
            "sequencia": item.sequencia,
            "inicio": inicio_prev,
            "fim": fim_prev
        })

        data_anterior = fim_prev

    ultimo = fila[-1]

    produto_reg = PRODUTOS_260610.objects.filter(
        peca=produto,
        operacao=operacao
    ).first()

    if not produto_reg:
        return render(request, "colab/grafico_4.html", {
            "grafico": None,
            "maquinas": maquinas,
            "operacoes": operacoes,
            "produtos": produtos,
            "maquina": maquina,
            "operacao": operacao,
            "produto": produto,
            "modelo": modelo,
            "quantidade": quantidade
        })

    pecas_hora_final = float(produto_reg.pecas_hora)

    duracao_padrao = quantidade / pecas_hora_final
    duracao_prevista = duracao_padrao / (oee_previsto / 100)

    inicio_novo = data_anterior or timezone.now()
    fim_novo = inicio_novo + timedelta(hours=duracao_prevista)

    previsoes.append({
        "sequencia": ultimo.sequencia + 1,
        "inicio": inicio_novo,
        "fim": fim_novo,
        "simulada": True
    })

    fig, ax = plt.subplots(figsize=(15, 6))

    for i, p in enumerate(previsoes):

        ini = mdates.date2num(p["inicio"])
        fim = mdates.date2num(p["fim"])
        largura = fim - ini

        cor = "#2E86DE" if not p.get("simulada") else "#27AE60"

        ax.barh(i, largura, left=ini, height=0.5, color=cor)

        ax.text(
            fim,
            i,
            p["fim"].strftime("%d/%m/%Y %H:%M"),
            va="center",
            fontsize=9
        )

    ax.set_yticks(range(len(previsoes)))
    ax.set_yticklabels([f"SEQ {p['sequencia']}" for p in previsoes])

    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    plt.xticks(rotation=25)

    plt.title("Previsão de Sequências")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()

    buffer.seek(0)
    grafico_png = base64.b64encode(buffer.getvalue()).decode()

    return render(request, "colab/grafico_4.html", {
        "grafico": grafico_png,
        "maquinas": maquinas,
        "operacoes": operacoes,
        "produtos": produtos,
        "maquina": maquina,
        "operacao": operacao,
        "produto": produto,
        "modelo": modelo,
        "quantidade": quantidade,
        "oee_previsto": round(oee_previsto, 2),
        "pecas_hora": round(pecas_hora_final, 2),
        "duracao_padrao": duracao_padrao,
        "duracao_prevista": duracao_prevista,
        "previsao_inicio": inicio_novo,
        "fim_previsto": fim_novo
    })








# @login_required
# def grafico_4(request):

#     maquinas = list(
#         ITENS_260611.objects
#         .exclude(maquina__isnull=True)
#         .values_list("maquina", flat=True)
#         .distinct()
#         .order_by("maquina")
#     )

#     maquina = request.GET.get("maquina")
#     operacao = request.GET.get("operacao")
#     produto = request.GET.get("produto")
#     quantidade = request.GET.get("quantidade")
#     modelo = request.GET.get("modelo", "")

#     operacoes = []
#     produtos = []

#     if maquina:
#         operacoes = list(
#             ITENS_260611.objects
#             .filter(maquina=int(maquina))
#             .exclude(operacao__isnull=True)
#             .exclude(operacao=0)
#             .values_list("operacao", flat=True)
#             .distinct()
#             .order_by("operacao")
#         )

#     produtos = list(
#         ITENS_260611.objects
#         .exclude(peca__isnull=True)
#         .values_list("peca", flat=True)
#         .distinct()
#         .order_by("peca")
#     )

#     if not operacao or not produto or not quantidade:
#         return render(request, "colab/grafico_4.html", {
#             "grafico": None,
#             "maquinas": maquinas,
#             "operacoes": operacoes,
#             "produtos": produtos,
#             "maquina": maquina,
#             "operacao": operacao,
#             "produto": produto,
#             "modelo": modelo,
#             "quantidade": quantidade,
#             "mensagem": "Preencha operação, produto e quantidade (máquina é opcional)."
#         })

#     maquina = int(maquina) if maquina else None
#     operacao = int(operacao)
#     produto = int(produto)
#     quantidade = int(quantidade)

#     fila_query = ITENS_260611.objects.filter(
#         operacao=operacao
#     ).exclude(operacao=0)

#     if maquina:
#         fila_query = fila_query.filter(maquina=maquina)

#     fila = list(fila_query.order_by("sequencia"))

#     if not fila:
#         return render(request, "colab/grafico_4.html", {
#             "mensagem": "Fila vazia (não existem ordens para operação selecionada)."
#         })

#     historico_oee = OEE_Prod_260611.objects.filter(
#         operacao=operacao,
#         produto=produto,
#         oee__isnull=False
#     )

#     if maquina:
#         historico_oee = historico_oee.filter(maquina=maquina)

#     historico_oee = historico_oee.order_by("inicio")

#     df_oee = pd.DataFrame(list(historico_oee.values("oee")))

#     if df_oee.empty:
#         return render(request, "colab/grafico_4.html", {
#             "mensagem": "Sem histórico de OEE."
#         })

#     serie_oee = df_oee["oee"].astype(float)

#     if len(serie_oee) == 1:
#         oee_previsto = float(serie_oee.iloc[0])
#     else:
#         X = np.arange(len(serie_oee)).reshape(-1, 1)
#         y = serie_oee.values.reshape(-1, 1)

#         if modelo == "arima":
#             oee_previsto = prever_arima_simples(serie_oee.values)

#         elif modelo == "svr":
#             scaler_x = StandardScaler()
#             scaler_y = StandardScaler()

#             Xs = scaler_x.fit_transform(X)
#             ys = scaler_y.fit_transform(y).ravel()

#             svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01)
#             svr.fit(Xs, ys)

#             pred = svr.predict(scaler_x.transform([[len(serie_oee)]]))
#             oee_previsto = scaler_y.inverse_transform([pred])[0][0]

#         else:
#             model = LinearRegression()
#             model.fit(X, serie_oee)
#             oee_previsto = model.predict([[len(serie_oee)]])[0]

#     oee_previsto = max(oee_previsto, 1)

#     previsoes = []
#     data_anterior = None

#     for idx, item in enumerate(fila):

#         quantidade_restante = max(
#             0,
#             (item.programado or 0) - (item.produzido or 0)
#         )

#         pecas_hora_item = float(item.pecas_hora or 1)

#         tempo_ideal = quantidade_restante / pecas_hora_item
#         tempo_real = tempo_ideal / (oee_previsto / 100)

#         if idx == 0:
#             inicio_prev = item.prev_inicio or item.inicio
#         else:
#             inicio_prev = data_anterior

#         if not inicio_prev:
#             continue

#         fim_prev = inicio_prev + timedelta(hours=tempo_real)

#         previsoes.append({
#             "sequencia": item.sequencia,
#             "inicio": inicio_prev,
#             "fim": fim_prev
#         })

#         data_anterior = fim_prev

#     ultimo = fila[-1]

#     produto_reg = PRODUTOS_260610.objects.filter(
#         peca=produto,
#         operacao=operacao
#     ).first()

#     if not produto_reg:
#         return render(request, "colab/grafico_4.html", {
#             "mensagem": "Produto não encontrado na base PRODUTOS."
#         })

#     pecas_hora_final = float(produto_reg.pecas_hora)

#     # =========================
#     # NOVOS CÁLCULOS (PEDIDO)
#     # =========================

#     duracao_padrao = quantidade / pecas_hora_final
#     duracao_prevista = duracao_padrao / (oee_previsto / 100)

#     inicio_novo = data_anterior or timezone.now()
#     fim_novo = inicio_novo + timedelta(hours=duracao_prevista)

#     previsoes.append({
#         "sequencia": ultimo.sequencia + 1,
#         "inicio": inicio_novo,
#         "fim": fim_novo,
#         "simulada": True
#     })

#     # GRÁFICO (igual)

#     fig, ax = plt.subplots(figsize=(15, 6))

#     for i, p in enumerate(previsoes):

#         ini = mdates.date2num(p["inicio"])
#         fim = mdates.date2num(p["fim"])
#         largura = fim - ini

#         cor = "#2E86DE" if not p.get("simulada") else "#27AE60"

#         ax.barh(i, largura, left=ini, height=0.5, color=cor)

#         ax.text(
#             fim,
#             i,
#             p["fim"].strftime("%d/%m/%Y %H:%M"),
#             va="center",
#             fontsize=9
#         )

#     ax.set_yticks(range(len(previsoes)))
#     ax.set_yticklabels([f"SEQ {p['sequencia']}" for p in previsoes])

#     ax.invert_yaxis()
#     ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
#     plt.xticks(rotation=25)

#     plt.title("Previsão de Sequências")
#     plt.tight_layout()

#     buffer = io.BytesIO()
#     plt.savefig(buffer, format="png")
#     plt.close()

#     buffer.seek(0)
#     grafico_png = base64.b64encode(buffer.getvalue()).decode()

#     return render(request, "colab/grafico_4.html", {
#         "grafico": grafico_png,
#         "maquinas": maquinas,
#         "operacoes": operacoes,
#         "produtos": produtos,
#         "maquina": maquina,
#         "operacao": operacao,
#         "produto": produto,
#         "modelo": modelo,
#         "quantidade": quantidade,
#         "oee_previsto": round(oee_previsto, 2),
#         "pecas_hora": round(pecas_hora_final, 2),

#         # NOVOS CAMPOS
#         "duracao_padrao": duracao_padrao,
#         "duracao_prevista": duracao_prevista,
#         "previsao_inicio": inicio_novo,

#         "fim_previsto": fim_novo
#     })










@login_required
def powerbi(request):
    powerbi_link = "https://app.powerbi.com/view?r=eyJrIjoiODgyZmU5ZDEtNWQxYi00N2YwLTk5ZWMtZDMwYmU1ZWE2ZjVhIiwidCI6ImNmNzJlMmJkLTdhMmItNDc4My1iZGViLTM5ZDU3YjA3Zjc2ZiIsImMiOjR9"
    return render(request, "colab/powerbi.html", {
        "powerbi_url": powerbi_link
    })

