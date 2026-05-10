from __future__ import annotations

from django.db import models


class Symbol(models.Model):
    id = models.BigAutoField(primary_key=True)
    exchange = models.CharField(max_length=8)    # NSE | BSE | NFO | MCX
    token = models.CharField(max_length=32)
    tradingsymbol = models.CharField(max_length=80)
    name = models.CharField(max_length=200, blank=True)
    segment = models.CharField(max_length=16, blank=True)
    lot_size = models.IntegerField(default=1)
    tick_size = models.DecimalField(max_digits=8, decimal_places=4, default=0.05)

    class Meta:
        unique_together = [("exchange", "token")]
        indexes = [models.Index(fields=["exchange", "tradingsymbol"])]


class Candle(models.Model):
    id = models.BigAutoField(primary_key=True)
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE, related_name="candles")
    interval = models.CharField(max_length=8)
    t = models.DateTimeField(db_index=True)
    o = models.DecimalField(max_digits=14, decimal_places=4)
    h = models.DecimalField(max_digits=14, decimal_places=4)
    l = models.DecimalField(max_digits=14, decimal_places=4)
    c = models.DecimalField(max_digits=14, decimal_places=4)
    v = models.BigIntegerField()

    class Meta:
        unique_together = [("symbol", "interval", "t")]
        indexes = [models.Index(fields=["symbol", "interval", "-t"])]
