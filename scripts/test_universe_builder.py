"""测试 UniverseBuilder 模块."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trader_assist.data_collector.pg_client import get_db
from ai_trader_assist.universe.builder import UniverseBuilder, UniverseConfig
from ai_trader_assist.risk_engine.market_regime import MarketRegime


def test_universe_builder():
    print("=" * 70)
    print("UniverseBuilder 测试")
    print("=" * 70)
    
    db = get_db()
    builder = UniverseBuilder(db=db)
    
    test_date = date(2025, 12, 15)
    
    for regime in [MarketRegime.BULL_TREND, MarketRegime.RANGE_BOUND, MarketRegime.BEAR_TREND]:
        print(f"\n【{regime.value}】")
        print("-" * 50)
        
        summary = builder.get_universe_summary(test_date, regime)
        
        print(f"总股票数: {summary['total_count']}")
        print(f"分层: Core={summary['by_pool'].get('core', 0)}, "
              f"Rotation={summary['by_pool'].get('rotation', 0)}, "
              f"Candidate={summary['by_pool'].get('candidate', 0)}")
        
        print(f"\n板块分布:")
        for sector, count in sorted(summary['by_sector'].items(), key=lambda x: -x[1]):
            print(f"  {sector}: {count}只")
        
        print(f"\nWatchlist ({len(summary['watchlist'])}只):")
        print(f"  {', '.join(summary['watchlist'][:20])}")
        if len(summary['watchlist']) > 20:
            print(f"  ... 及其他 {len(summary['watchlist']) - 20} 只")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_universe_builder()
