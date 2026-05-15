"""Phase 4c — trading app, part 1: the portfolio + orders models.

Consolidates apps.portfolio (0001 + 0002) and apps.orders (0001) into the
new `trading` app. Tables keep their original names via db_table, so
existing databases need no DDL — only the django_migrations bookkeeping
moves (see the phase-4c commit message for the recorder reconciliation).

Split into two migrations because Trade/OptionsPosition FK agents_core,
which itself FKs trading.Portfolio — 0002 carries those.
"""
import django.db.models.deletion
import uuid
from decimal import Decimal
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('market_data', '0001_initial'),
        ('tenants', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Portfolio',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(default='Default', max_length=80)),
                ('capital', models.DecimalField(decimal_places=2, default=Decimal('0'), max_digits=16)),
                ('used_capital', models.DecimalField(decimal_places=2, default=Decimal('0'), max_digits=16)),
                ('realized_pnl', models.DecimalField(decimal_places=2, default=Decimal('0'), max_digits=16)),
                ('day_pnl', models.DecimalField(decimal_places=2, default=Decimal('0'), max_digits=16)),
                ('mode', models.CharField(default='paper', max_length=10)),
                ('broker_link', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='portfolios', to='market_data.brokerlink')),
                ('tenant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='+', to='tenants.tenant')),
            ],
            options={
                'db_table': 'portfolio_portfolio',
            },
        ),
        migrations.CreateModel(
            name='PortfolioSnapshot',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('captured_at', models.DateTimeField(auto_now_add=True, db_index=True)),
                ('equity', models.DecimalField(decimal_places=2, max_digits=16)),
                ('day_pnl', models.DecimalField(decimal_places=2, max_digits=16)),
                ('unrealized_pnl', models.DecimalField(decimal_places=2, max_digits=16)),
                ('open_positions', models.IntegerField()),
                ('portfolio', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='snapshots', to='trading.portfolio')),
                ('tenant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='+', to='tenants.tenant')),
            ],
            options={
                'db_table': 'portfolio_portfoliosnapshot',
            },
        ),
        migrations.CreateModel(
            name='Position',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('symbol', models.CharField(max_length=80)),
                ('side', models.CharField(max_length=4)),
                ('qty', models.IntegerField()),
                ('avg_price', models.DecimalField(decimal_places=4, max_digits=14)),
                ('last_ltp', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('sl', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('tp', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('unrealized_pnl', models.DecimalField(decimal_places=2, default=Decimal('0'), max_digits=16)),
                ('exit_price', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('realized_pnl', models.DecimalField(decimal_places=2, default=Decimal('0'), max_digits=16)),
                ('exchange', models.CharField(default='NSE', max_length=8)),
                ('status', models.CharField(choices=[('open', 'Open'), ('closed', 'Closed')], default='open', max_length=10)),
                ('opened_at', models.DateTimeField(auto_now_add=True)),
                ('closed_at', models.DateTimeField(blank=True, null=True)),
                ('portfolio', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='positions', to='trading.portfolio')),
                ('tenant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='+', to='tenants.tenant')),
            ],
            options={
                'db_table': 'portfolio_position',
            },
        ),
        migrations.CreateModel(
            name='Order',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('symbol', models.CharField(max_length=80)),
                ('side', models.CharField(choices=[('BUY', 'Buy'), ('SELL', 'Sell')], max_length=4)),
                ('qty', models.IntegerField()),
                ('order_type', models.CharField(default='MARKET', max_length=16)),
                ('product', models.CharField(choices=[('INTRADAY', 'Intraday'), ('DELIVERY', 'Delivery'), ('CARRYFORWARD', 'Carryforward')], default='INTRADAY', max_length=16)),
                ('price', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('sl', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('tp', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('broker_order_id', models.CharField(blank=True, max_length=80)),
                ('status', models.CharField(choices=[('queued', 'Queued'), ('sent', 'Sent'), ('open', 'Open'), ('filled', 'Filled'), ('cancelled', 'Cancelled'), ('rejected', 'Rejected'), ('failed', 'Failed')], default='queued', max_length=16)),
                ('idempotency_key', models.CharField(blank=True, db_index=True, max_length=128)),
                ('origin', models.CharField(default='ui', max_length=32)),
                ('error', models.TextField(blank=True)),
                ('broker_link', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='market_data.brokerlink')),
                ('created_by', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to=settings.AUTH_USER_MODEL)),
                ('portfolio', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='trading.portfolio')),
                ('tenant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='+', to='tenants.tenant')),
            ],
            options={
                'db_table': 'orders_order',
            },
        ),
        migrations.CreateModel(
            name='OutboxEvent',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('payload', models.JSONField()),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('in_flight', 'In Flight'), ('succeeded', 'Succeeded'), ('failed', 'Failed'), ('dlq', 'Dlq')], default='pending', max_length=16)),
                ('attempts', models.IntegerField(default=0)),
                ('last_error', models.TextField(blank=True)),
                ('next_run_at', models.DateTimeField(auto_now_add=True, db_index=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('order', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='events', to='trading.order')),
            ],
            options={
                'db_table': 'orders_outboxevent',
            },
        ),
        migrations.AddIndex(
            model_name='portfolio',
            index=models.Index(fields=['tenant', 'mode'], name='portfolio_p_tenant__edc91b_idx'),
        ),
        migrations.AddIndex(
            model_name='position',
            index=models.Index(fields=['tenant', 'portfolio', 'status'], name='portfolio_p_tenant__a3ca52_idx'),
        ),
        migrations.AddIndex(
            model_name='position',
            index=models.Index(fields=['tenant', 'symbol'], name='portfolio_p_tenant__130ef3_idx'),
        ),
        migrations.AddIndex(
            model_name='position',
            index=models.Index(fields=['tenant', 'portfolio', 'opened_at'], name='portfolio_p_tenant__9d5754_idx'),
        ),
        migrations.AddIndex(
            model_name='order',
            index=models.Index(fields=['tenant', 'status', '-created_at'], name='orders_orde_tenant__1d191c_idx'),
        ),
        migrations.AddIndex(
            model_name='order',
            index=models.Index(fields=['tenant', 'portfolio', '-created_at'], name='orders_orde_tenant__34015e_idx'),
        ),
        migrations.AddIndex(
            model_name='outboxevent',
            index=models.Index(fields=['status', 'next_run_at'], name='orders_outb_status_7ffd84_idx'),
        ),
    ]
