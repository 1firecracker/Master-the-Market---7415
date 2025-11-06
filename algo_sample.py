from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from datetime import datetime, timedelta
import statsmodels.api as sm

class AlgoEvent:
    def __init__(self):
        self.lasttradetime = datetime(2000,1,1)
        self.orderPairCnt = 0 
        self.arrSize = 5
        self.arr_closeY = []
        self.arr_closeX = []

    def start(self, mEvt):
        self.myinstrument_Y = mEvt['subscribeList'][0]
        self.myinstrument_X = mEvt['subscribeList'][1]
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        self.evt.start()

    def on_bulkdatafeed(self, isSync, bd, ab):
        if isSync:
            # check condition for open position
            if bd[self.myinstrument_Y]['timestamp'] >= self.lasttradetime + timedelta(hours=24):
                self.lasttradetime = bd[self.myinstrument_Y]['timestamp']
                # collect observations
                self.arr_closeY.append(bd[self.myinstrument_Y]['lastPrice'])
                self.arr_closeX.append(bd[self.myinstrument_X]['lastPrice'])
                # kick out the oldest observation if array size is too long
                if len(self.arr_closeY)>self.arrSize:
                    self.arr_closeY = self.arr_closeY[-self.arrSize:]
                if len(self.arr_closeX)>self.arrSize:
                    self.arr_closeX = self.arr_closeX[-self.arrSize:]
                # fit linear regression
                Y = self.arr_closeY
                X = self.arr_closeX
                #X = sm.add_constant(X)   #add this line if you want to include intercept in the regression
                model = sm.OLS(Y, X)
                results = model.fit()
                self.evt.consoleLog(results.summary())
                coeff_b, tvalue, mse = results.params[-1], results.tvalues, results.mse_resid
                # compute current residual, e = Y - b*X
                diff = self.arr_closeY[-1] - coeff_b*self.arr_closeX[-1]
                
                if diff>0.1*mse:  # regard Y as overpriced, X as underpriced
                    self.orderPairCnt += 1
                    self.openOrder(-1, self.myinstrument_Y, self.orderPairCnt, 1)  #short Y
                    if coeff_b>0:
                        self.openOrder(1, self.myinstrument_X, self.orderPairCnt, abs(round(coeff_b,2)))   #long X
                    else:
                        self.openOrder(-1, self.myinstrument_X, self.orderPairCnt, abs(round(coeff_b,2)))   #short X
                elif diff<-0.1*mse:  # regard Y as underpriced, X as overpriced
                    self.orderPairCnt += 1
                    self.openOrder(1, self.myinstrument_Y, self.orderPairCnt, 1)  #long Y
                    if coeff_b>0:
                        self.openOrder(-1, self.myinstrument_X, self.orderPairCnt, abs(round(coeff_b,2)))   #short X
                    else:
                        self.openOrder(1, self.myinstrument_X, self.orderPairCnt, abs(round(coeff_b,2)))   #long X

    def openOrder(self, buysell, instrument, orderRef, volume):
        order = AlgoAPIUtil.OrderObject()
        order.instrument = instrument
        order.orderRef = orderRef
        order.volume = volume
        order.openclose = 'open'
        order.buysell = buysell
        order.ordertype = 0 #0=market_order, 1=limit_order
        order.holdtime = self.arrSize*24*60*60 #unit in second
        self.evt.sendOrder(order)

    def on_marketdatafeed(self, md, ab):
        pass

    def on_orderfeed(self, of):
        pass

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        pass
