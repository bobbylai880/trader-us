"""V8.2 风控模块 - 预防式4级状态机"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional


@dataclass
class RiskState:
    """风控状态"""
    date: str
    vix: float
    vix_sma5: float
    vix_sma20: float
    vix_trend: str  # rising/falling/stable
    spy_below_sma50: bool
    spy_momentum: float
    mode: str  # normal/watch/caution/danger
    max_exposure: float
    cooldown: int
    reason: str


class RiskControl:
    """预防式风控 - 4级状态机
    
    状态转换逻辑：
    - 升级快：一旦触发更严重的条件，立即升级并设置冷却期
    - 降级慢：必须等冷却期结束才能降级
    
    | Mode    | Max Exp | Cooldown | 触发条件 |
    |---------|---------|----------|----------|
    | Normal  | 95%     | 0        | 默认状态 |
    | Watch   | 70%     | 3天      | VIX>20且上升 或 SPY动量<-8% |
    | Caution | 50%     | 5天      | VIX>22且上升 且 SPY<SMA50 |
    | Danger  | 30%     | 10天     | VIX>28 或 VIX急升>30% |
    """
    
    MODE_CONFIG = {
        "normal":  {"exposure": 0.95, "cooldown": 0,  "severity": 0},
        "watch":   {"exposure": 0.70, "cooldown": 3,  "severity": 1},
        "caution": {"exposure": 0.50, "cooldown": 5,  "severity": 2},
        "danger":  {"exposure": 0.30, "cooldown": 10, "severity": 3},
    }
    
    def __init__(self):
        self._mode = "normal"
        self._cooldown = 0
    
    def check(
        self,
        dt: date,
        get_val: Callable[[str, date, str], Optional[float]],
    ) -> RiskState:
        """检查风控状态
        
        Args:
            dt: 当前日期
            get_val: 数据获取函数 (symbol, date, column) -> value
        """
        # 获取指标
        vix = get_val("VIX", dt, "close") or 20
        vix_sma5 = get_val("VIX", dt, "sma5") or vix
        vix_sma20 = get_val("VIX", dt, "sma20") or vix
        vix_mom5 = get_val("VIX", dt, "mom5") or 0
        
        spy_close = get_val("SPY", dt, "close") or 0
        spy_sma50 = get_val("SPY", dt, "sma50") or spy_close
        spy_mom = get_val("SPY", dt, "mom20") or 0
        
        spy_below_sma50 = spy_close < spy_sma50
        
        # VIX趋势判断
        if vix_sma5 > vix_sma20 * 1.1:
            vix_trend = "rising"
        elif vix_sma5 < vix_sma20 * 0.9:
            vix_trend = "falling"
        else:
            vix_trend = "stable"
        
        # 冷却期处理
        if self._cooldown > 0:
            self._cooldown -= 1
            return RiskState(
                date=str(dt), vix=vix, vix_sma5=vix_sma5, vix_sma20=vix_sma20,
                vix_trend=vix_trend, spy_below_sma50=spy_below_sma50,
                spy_momentum=spy_mom, mode=self._mode,
                max_exposure=self.MODE_CONFIG[self._mode]["exposure"],
                cooldown=self._cooldown,
                reason=f"冷却期({self._cooldown}天)"
            )
        
        # 判断新状态
        if vix > 28 or (vix > 22 and vix_mom5 > 0.3):
            new_mode = "danger"
            reason = f"VIX高位({vix:.1f})" if vix > 28 else f"VIX急升({vix_mom5:.1%})"
        elif vix > 22 and vix_trend == "rising" and spy_below_sma50:
            new_mode = "caution"
            reason = f"VIX上升({vix:.1f}) + SPY破位"
        elif vix > 20 and vix_trend == "rising":
            new_mode = "watch"
            reason = f"VIX上升趋势({vix:.1f})"
        elif spy_mom < -0.08:
            new_mode = "watch"
            reason = f"市场回调({spy_mom:.1%})"
        else:
            new_mode = "normal"
            reason = "正常运行"
        
        # 状态转换
        new_severity = self.MODE_CONFIG[new_mode]["severity"]
        old_severity = self.MODE_CONFIG[self._mode]["severity"]
        
        if new_severity > old_severity:
            # 升级：立即生效 + 设置冷却
            self._mode = new_mode
            self._cooldown = self.MODE_CONFIG[new_mode]["cooldown"]
        elif new_severity < old_severity and self._cooldown == 0:
            # 降级：仅在冷却结束后
            self._mode = new_mode
        
        return RiskState(
            date=str(dt), vix=vix, vix_sma5=vix_sma5, vix_sma20=vix_sma20,
            vix_trend=vix_trend, spy_below_sma50=spy_below_sma50,
            spy_momentum=spy_mom, mode=self._mode,
            max_exposure=self.MODE_CONFIG[self._mode]["exposure"],
            cooldown=self._cooldown, reason=reason
        )
    
    def reset(self):
        """重置状态"""
        self._mode = "normal"
        self._cooldown = 0
