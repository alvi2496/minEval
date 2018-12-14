import data.processor as processor
import paradigms.verification.controller as verifier
import warnings

warnings.filterwarnings("ignore")

data_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSGhxSbBeeXdkRdBVQ9wSL1aJTs52SXV3NKfcfoX1wI89XDCJMC5tW0HZk5HYdh2xT0DtufMLSn9hHX/pub?gid=1193567183&single=true&output=csv"

data = processor.process(data_url)

verifier.verify(data, 0.3, 10)
