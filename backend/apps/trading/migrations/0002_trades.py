"""Phase 4c — trading app, part 2: the trades models.

Trade + OptionsPosition both FK agents_core.AgentRun, and AgentRun FKs
trading.Portfolio (from 0001) — so these models land in a second
migration that depends on agents_core/0001, breaking the cycle.

Consolidates apps.trades (0001). Tables keep their original names.
"""
import django.db.models.deletion
import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('trading', '0001_initial'),
        ('agents_core', '0001_initial'),
        ('tenants', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='OptionsPosition',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('position_type', models.CharField(choices=[('SHORT_STRADDLE', 'Short Straddle'), ('SHORT_STRANGLE', 'Short Strangle'), ('LONG_STRADDLE', 'Long Straddle'), ('BULL_CALL_SPREAD', 'Bull Call Spread'), ('BEAR_PUT_SPREAD', 'Bear Put Spread'), ('IRON_CONDOR', 'Iron Condor'), ('PYRAMID_OPTION', 'Pyramid Option'), ('CUSTOM', 'Custom')], max_length=24)),
                ('underlying', models.CharField(db_index=True, max_length=20)),
                ('expiry', models.DateField()),
                ('lot_size', models.IntegerField(default=1)),
                ('lots', models.IntegerField(default=1)),
                ('status', models.CharField(choices=[('ACTIVE', 'Active'), ('PARTIAL', 'Partial'), ('HEDGED', 'Hedged'), ('CLOSED', 'Closed')], db_index=True, default='ACTIVE', max_length=10)),
                ('net_delta', models.DecimalField(decimal_places=4, default=0, max_digits=8)),
                ('current_pnl_inr', models.DecimalField(decimal_places=2, default=0, max_digits=16)),
                ('realized_pnl', models.DecimalField(decimal_places=2, default=0, max_digits=16)),
                ('opened_at', models.DateTimeField(blank=True, null=True)),
                ('closed_at', models.DateTimeField(blank=True, null=True)),
                ('trade_date', models.DateField(db_index=True)),
                ('legacy_straddle_id', models.IntegerField(blank=True, db_index=True, null=True)),
                ('portfolio', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='options_positions', to='trading.portfolio')),
                ('strategy_run', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to='agents_core.agentrun')),
                ('tenant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='+', to='tenants.tenant')),
            ],
            options={
                'db_table': 'trades_optionsposition',
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='OptionsLeg',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('leg_role', models.CharField(choices=[('SHORT_CE', 'Short Ce'), ('SHORT_PE', 'Short Pe'), ('LONG_CE', 'Long Ce'), ('LONG_PE', 'Long Pe'), ('HEDGE_FUT', 'Hedge Fut')], max_length=12)),
                ('symbol', models.CharField(max_length=40)),
                ('token', models.CharField(blank=True, default='', help_text='Broker token (Angel One NFO/BFO)', max_length=20)),
                ('strike', models.IntegerField(blank=True, null=True)),
                ('qty', models.IntegerField(help_text='Absolute lots × lot_size; sign comes from leg_role')),
                ('open_price', models.DecimalField(decimal_places=4, max_digits=14)),
                ('current_price', models.DecimalField(decimal_places=4, default=0, max_digits=14)),
                ('closed_price', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('closed_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('position', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='legs', to='trading.optionsposition')),
            ],
            options={
                'db_table': 'trades_optionsleg',
                'ordering': ['id'],
            },
        ),
        migrations.CreateModel(
            name='Trade',
            fields=[
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('symbol', models.CharField(db_index=True, max_length=80)),
                ('exchange', models.CharField(default='NSE', max_length=8)),
                ('side', models.CharField(choices=[('BUY', 'Buy'), ('SELL', 'Sell')], max_length=4)),
                ('product', models.CharField(default='INTRADAY', help_text='INTRADAY | DELIVERY | CARRYFORWARD', max_length=16)),
                ('entry_price', models.DecimalField(decimal_places=4, max_digits=14)),
                ('stop_loss', models.DecimalField(decimal_places=4, max_digits=14)),
                ('target', models.DecimalField(decimal_places=4, max_digits=14)),
                ('quantity', models.IntegerField()),
                ('lot_size', models.IntegerField(default=1)),
                ('confidence', models.DecimalField(decimal_places=4, default=0, max_digits=5)),
                ('reasoning', models.TextField(blank=True, default='', help_text='LLM rationale; empty for non-LLM workflows')),
                ('risk_approved', models.BooleanField(default=False)),
                ('risk_reason', models.CharField(blank=True, default='', max_length=255)),
                ('risk_details', models.JSONField(blank=True, default=dict, help_text='Full 10-criterion result for the audit')),
                ('risk_decided_at', models.DateTimeField(blank=True, null=True)),
                ('status', models.CharField(choices=[('PLAN', 'Plan'), ('APPROVED', 'Approved'), ('REJECTED', 'Rejected'), ('QUEUED', 'Queued'), ('SENT', 'Sent'), ('PARTIAL', 'Partial'), ('FILLED', 'Filled'), ('CLOSED', 'Closed'), ('CANCELLED', 'Cancelled'), ('EXPIRED', 'Expired')], db_index=True, default='PLAN', max_length=16)),
                ('fill_price', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('fill_quantity', models.IntegerField(blank=True, null=True)),
                ('filled_at', models.DateTimeField(blank=True, null=True)),
                ('exit_price', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('exit_quantity', models.IntegerField(blank=True, null=True)),
                ('closed_at', models.DateTimeField(blank=True, null=True)),
                ('close_reason', models.CharField(blank=True, choices=[('SL_HIT', 'Sl Hit'), ('TARGET_HIT', 'Target Hit'), ('MANUAL', 'Manual'), ('EOD', 'Eod'), ('TRAIL', 'Trail')], default='', max_length=32)),
                ('realized_pnl', models.DecimalField(blank=True, decimal_places=2, max_digits=16, null=True)),
                ('unrealized_pnl', models.DecimalField(decimal_places=2, default=0, max_digits=16)),
                ('pnl_percent', models.DecimalField(blank=True, decimal_places=4, max_digits=8, null=True)),
                ('last_ltp', models.DecimalField(blank=True, decimal_places=4, max_digits=14, null=True)),
                ('origin', models.CharField(choices=[('workflow', 'Workflow'), ('manual', 'Manual'), ('api', 'Api'), ('broker_sync', 'Broker Sync')], default='workflow', max_length=16)),
                ('trade_date', models.DateField(db_index=True)),
                ('legacy_trade_journal_id', models.IntegerField(blank=True, db_index=True, null=True)),
                ('portfolio', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='trades', to='trading.portfolio')),
                ('primary_order', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to='trading.order')),
                ('strategy_run', models.ForeignKey(blank=True, help_text='Workflow run that produced this trade', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='trades', to='agents_core.agentrun')),
                ('tenant', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='+', to='tenants.tenant')),
            ],
            options={
                'db_table': 'trades_trade',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='optionsposition',
            index=models.Index(fields=['tenant', 'status'], name='trades_opti_tenant__4596a5_idx'),
        ),
        migrations.AddIndex(
            model_name='optionsposition',
            index=models.Index(fields=['tenant', 'underlying', 'expiry'], name='trades_opti_tenant__75fd1f_idx'),
        ),
        migrations.AddIndex(
            model_name='optionsleg',
            index=models.Index(fields=['position', 'leg_role'], name='trades_opti_positio_2212db_idx'),
        ),
        migrations.AddIndex(
            model_name='trade',
            index=models.Index(fields=['tenant', 'status', '-created_at'], name='trades_trad_tenant__b7e3d2_idx'),
        ),
        migrations.AddIndex(
            model_name='trade',
            index=models.Index(fields=['tenant', 'portfolio', 'status'], name='trades_trad_tenant__fce6c9_idx'),
        ),
        migrations.AddIndex(
            model_name='trade',
            index=models.Index(fields=['tenant', 'symbol', 'trade_date'], name='trades_trad_tenant__f103db_idx'),
        ),
        migrations.AddIndex(
            model_name='trade',
            index=models.Index(fields=['tenant', 'strategy_run'], name='trades_trad_tenant__237bdf_idx'),
        ),
    ]
