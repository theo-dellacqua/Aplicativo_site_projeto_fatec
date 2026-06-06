from django.db import models


class OEE_Prod_260521(models.Model):
    recid = models.IntegerField(primary_key=True)
    maquina = models.IntegerField(null=True)
    inicio = models.DateTimeField(null=True)
    os = models.BigIntegerField(null=True)
    produto = models.IntegerField(null=True)
    operacao = models.IntegerField(null=True)
    molde = models.IntegerField(null=True)
    ttotal = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    sec_total = models.IntegerField(null=True)
    tdisp = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    sec_disp = models.IntegerField(null=True)
    tprod = models.DecimalField(max_digits=10, decimal_places=4, null=True)
    sec_prod = models.IntegerField(null=True)
    qtde_std = models.IntegerField(null=True)
    qtde_real = models.IntegerField(null=True)
    qtde_boas = models.IntegerField(null=True)
    teep = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    oee = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    disp = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    perf = models.DecimalField(max_digits=6, decimal_places=2, null=True)
    qual = models.DecimalField(max_digits=6, decimal_places=2, null=True)


    def __str__(self):
        return f'{self.recid} {self.maquina} {self.inicio} {self.os} {self.produto}'
    


class ITENS_260521(models.Model):
    recid = models.IntegerField(primary_key=True)
    os = models.BigIntegerField(null=True)
    maquina = models.IntegerField(null=True)
    sequencia = models.IntegerField(null=True)
    operacao = models.IntegerField(null=True)
    peca = models.IntegerField(null=True)
    molde = models.IntegerField(null=True)
    status = models.IntegerField(null=True)
    programacao = models.DateTimeField(null=True)
    modificacao = models.DateTimeField(null=True)
    inicio = models.DateTimeField(null=True, blank=True)
    final = models.DateTimeField(null=True, blank=True)
    programado = models.IntegerField(null=True)
    produzido = models.IntegerField(null=True)
    rejeitado = models.IntegerField(null=True)
    pecas_hora = models.DecimalField(max_digits=10, decimal_places=2, null=True)
    production_order = models.BigIntegerField(null=True)
    itens_prd = models.IntegerField(null=True)
    itens_rej = models.IntegerField(null=True)
    lim_entrega = models.DateTimeField(null=True, blank=True)
    prev_inicio = models.DateTimeField(null=True, blank=True)
    prev_entrega = models.DateTimeField(null=True,blank=True)


    def __str__(self):
        return f'{self.recid} {self.os} {self.maquina} {self.peca}'