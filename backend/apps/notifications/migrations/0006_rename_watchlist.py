# Rename TradingViewWatchlist → Watchlist as a true RenameModel (preserves
# the underlying SQL table contents). Django's auto-detector emitted
# DELETE+CREATE because the model class moved within the same migration
# diff; this hand-written RenameModel is the right operation.
#
# The `related_name='watchlists'` change (was 'tradingview_watchlists') on
# the owner FK is reverse-only — it doesn't require DDL. Same for the
# constraint/index name changes Django would auto-emit; we let the next
# migration (0007 below) sweep those up after the rename completes.
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("notifications", "0005_tradingviewlink_watchlist"),
    ]

    operations = [
        migrations.RenameModel(
            old_name="TradingViewWatchlist",
            new_name="Watchlist",
        ),
    ]
