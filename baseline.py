from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import statsmodels.api as sm


class AlgoEvent:
    def __init__(self):
        self.lasttradetime = datetime(2000, 1, 1)
        self.orderPairCnt = 0
        self.arrSize = 5
        self.arr_closeY = []

    def start(self, mEvt):
        # 单标的版本：只订阅一个标的作为 Y
        self.myinstrument_Y = mEvt['subscribeList'][0]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()

    def on_bulkdatafeed(self, isSync, bd, ab):
        if isSync:
            # 每日（24小时）收集一次观测
            if bd[self.myinstrument_Y]['timestamp'] >= self.lasttradetime + timedelta(hours=24):
                self.lasttradetime = bd[self.myinstrument_Y]['timestamp']

                # 收集收盘价（或最后成交价）
                self.arr_closeY.append(bd[self.myinstrument_Y]['lastPrice'])

                # 滚动窗口
                if len(self.arr_closeY) > self.arrSize:
                    self.arr_closeY = self.arr_closeY[-self.arrSize:]

                # 样本不足则跳过
                if len(self.arr_closeY) < 2:
                    return

                # 单维均值回归：Y ~ const（等价于估计均值）
                Y = self.arr_closeY
                X = [1.0] * len(Y)
                model = sm.OLS(Y, X)
                results = model.fit()
                self.evt.consoleLog(results.summary())

                mean_level = results.params[-1]  # 常数项 = 均值
                mse = results.mse_resid

                # 当前偏离：e_t = Y_t - mean
                diff = self.arr_closeY[-1] - mean_level

                # 信号：偏离阈值（与原示例一致，使用 0.1 * mse）
                if diff > 0.1 * mse:  # 价格高于均值，做空
                    self.orderPairCnt += 1
                    self.openOrder(-1, self.myinstrument_Y, self.orderPairCnt, 1)
                elif diff < -0.1 * mse:  # 价格低于均值，做多
                    self.orderPairCnt += 1
                    self.openOrder(1, self.myinstrument_Y, self.orderPairCnt, 1)

    def openOrder(self, buysell, instrument, orderRef, volume):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = instrument
        order.orderRef = orderRef
        order.volume = volume
        order.openclose = 'open'
        order.buysell = buysell
        order.ordertype = 0  # 0=market_order, 1=limit_order
        order.holdtime = self.arrSize * 24 * 60 * 60  # 单位：秒
        self.evt.sendOrder(order)

    def on_marketdatafeed(self, md, ab):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass


