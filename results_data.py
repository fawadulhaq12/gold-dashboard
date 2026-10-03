"""Results of the research runs (2019-2024 walk-forward tests), copied from the Colab output.
The 2025-2026 period is locked and not used anywhere in this app."""
import pandas as pd

ALWAYS_UP_ACC = 0.543

ABLATION = pd.DataFrame(
    [("price", "logistic", .521, .510, .021, .516, .958, .157),
     ("price", "grad. boosting", .515, .514, .028, .525, .986, -.328),
     ("price + macro", "logistic", .536, .533, .065, .535, .706, .653),
     ("price + macro", "grad. boosting", .488, .493, -.015, .495, 1.000, -.410),
     ("price + news", "logistic", .494, .494, -.011, .510, 1.000, -.383),
     ("price + news", "grad. boosting", .510, .509, .018, .514, .995, -.036),
     ("price + macro + news", "logistic", .520, .519, .038, .528, .967, .414),
     ("price + macro + news", "grad. boosting", .504, .506, .012, .505, .999, -.284)],
    columns=["Features", "Model", "Accuracy", "Balanced acc.", "MCC", "AUC", "p vs always-up", "Sharpe"])

_VOL_COLS = ["Features", "Model", "RMSE", "R2 vs average", "R2 vs HAR", "DM p-value"]
VOL_ORIGINAL = pd.DataFrame(
    [("HAR", "ridge", .7426, .4905, .0000, None), ("HAR", "grad. boosting", .7303, .5073, .0330, .1335),
     ("HAR + VIX", "ridge", .7378, .4971, .0129, .0562), ("HAR + VIX", "grad. boosting", .7371, .4980, .0147, .3430),
     ("HAR + news", "ridge", .7660, .4579, -.0640, 1.000), ("HAR + news", "grad. boosting", .7682, .4548, -.0701, .9832),
     ("HAR + VIX + news", "ridge", .7608, .4653, -.0495, .9995),
     ("HAR + VIX + news", "grad. boosting", .7697, .4527, -.0743, .9810)], columns=_VOL_COLS)
VOL_CLEAN = pd.DataFrame(
    [("HAR", "ridge", .5367, .4542, .0000, None), ("HAR", "grad. boosting", .5458, .4356, -.0341, .8876),
     ("HAR + VIX", "ridge", .5360, .4557, .0028, .4017), ("HAR + VIX", "grad. boosting", .5473, .4325, -.0397, .9584),
     ("HAR + news", "ridge", .5504, .4261, -.0515, 1.000), ("HAR + news", "grad. boosting", .5756, .3723, -.1500, 1.000),
     ("HAR + VIX + news", "ridge", .5472, .4327, -.0393, .9943),
     ("HAR + VIX + news", "grad. boosting", .5681, .3885, -.1204, 1.000)], columns=_VOL_COLS)

EVENT_STUDY = pd.DataFrame(
    [("Russia-Ukraine", 41, .103, .533, .425, .281, .724, .766, .784),
     ("Middle East", 44, .368, .020, .209, .579, .749, .769, .631),
     ("India-Pakistan", 45, .042, .794, .238, .525, .719, .772, .840),
     ("US-Iran", 48, .034, .829, .162, .650, .777, .771, .430),
     ("Global conflict", 44, .234, .146, .139, .712, .730, .769, .768)],
    columns=["Hotspot", "Events", "Day-0 move", "p (day 0)", "Net move days 1-5", "p (net move)",
             "Size of moves", "Normal size", "p (size)"])

FAMOUS = pd.DataFrame(
    [("India strikes Balakot", "2019-02-26", 4.39, -0.35, -7.13, 1.43),
     ("Soleimani killed", "2020-01-03", 7.08, 2.57, 0.63, 1.00),
     ("Russia invades Ukraine", "2022-02-24", 5.78, 0.90, -0.08, 1.68),
     ("Hamas attack on Israel", "2023-10-09", 7.54, 2.00, 7.41, 1.76),
     ("Iran strikes Israel", "2024-04-15", 6.15, 0.13, -4.13, 1.53)],
    columns=["Event", "Trading day", "News spike (z)", "Gold day 0", "Gold net days 1-5", "Gold size days 1-5"])
