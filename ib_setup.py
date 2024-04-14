from ib_insync import *

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=1)

contract = Stock("AMD", "SMART", "USD")
ib.qualifyContracts(contract)
ticks = ib.reqHistoricalTicks(
    contract,
    startDateTime="20240322 16:00:00 US/Eastern",
    endDateTime="",
    numberOfTicks="1000",
    whatToShow="TRADES",
    useRth=False,
)

# convert to dataframe:
df = util.df(ticks)
print(df)
