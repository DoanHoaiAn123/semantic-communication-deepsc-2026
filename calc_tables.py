import sys
sys.stdout.reconfigure(encoding='utf-8')

def delta(val, ref): return round(val - ref, 2)
def recovery(att, def_, clean): return round((def_ - att) / (clean - att) * 100, 2)

# ============================================================
# BANG 5.7: Delta table cho TAP 1 (harouzie)
# Source: Sheet Bang_5.6_Tap1_harouzie
# ============================================================
print('=' * 65)
print('BANG 5.7 - Tap 1 (harouzie) - So sanh voi Clean')
print('=' * 65)

clean1    = {'acc': 71.59, 'f1': 71.59, 'bleu': 41.48}
fgsm1     = {'acc': 41.41, 'f1': 41.41, 'bleu': 14.02}
fgsm_def1 = {'acc': 64.48, 'f1': 64.48, 'bleu': 32.49}
cw1       = {'acc': 34.18, 'f1': 34.18, 'bleu':  7.97}
cw_def1   = {'acc': 60.57, 'f1': 60.57, 'bleu': 27.30}

rows1 = [
    ('FGSM (Chua phong thu)',   fgsm1,     None),
    ('FGSM + Co phong thu',     fgsm_def1, fgsm1),
    ('C+W (Chua phong thu)',    cw1,       None),
    ('C+W + Co phong thu',      cw_def1,   cw1),
]

print(f'\n{"Kich ban":<28} {"DAcc":>9} {"DF1":>9} {"DBLEU":>9} {"PhucHoi":>12}')
print('-' * 70)
for name, row, att in rows1:
    da = delta(row['acc'],  clean1['acc'])
    df = delta(row['f1'],   clean1['f1'])
    db = delta(row['bleu'], clean1['bleu'])
    if att is not None:
        rec = recovery(att['acc'], row['acc'], clean1['acc'])
        rec_s = f'{rec}%'
        # explain
        print(f'\n  [{name}]')
        print(f'    DAcc  = {row["acc"]}% - {clean1["acc"]}% = {da}%')
        print(f'    DF1   = {row["f1"]}% - {clean1["f1"]}% = {df}%')
        print(f'    DBLEU = {row["bleu"]}% - {clean1["bleu"]}% = {db}%')
        print(f'    PhucHoi = ({row["acc"]} - {att["acc"]}) / ({clean1["acc"]} - {att["acc"]}) x 100 = {rec}%')
    else:
        rec_s = '—'
        print(f'\n  [{name}]')
        print(f'    DAcc  = {row["acc"]}% - {clean1["acc"]}% = {da}%')
        print(f'    DF1   = {row["f1"]}% - {clean1["f1"]}% = {df}%')
        print(f'    DBLEU = {row["bleu"]}% - {clean1["bleu"]}% = {db}%')
        print(f'    PhucHoi = — (chua phong thu, khong ap dung)')

print()
print('=== KET QUA BANG 5.7 ===')
print(f'{"Kich ban":<28} {"DAcc":>9} {"DF1":>9} {"DBLEU":>9} {"PhucHoi":>12}')
print('-' * 70)
for name, row, att in rows1:
    da = delta(row['acc'],  clean1['acc'])
    df = delta(row['f1'],   clean1['f1'])
    db = delta(row['bleu'], clean1['bleu'])
    rec_s = f'{recovery(att["acc"], row["acc"], clean1["acc"])}%' if att else '—'
    print(f'{name:<28} {str(da)+"%":>9} {str(df)+"%":>9} {str(db)+"%":>9} {rec_s:>12}')

# ============================================================
# BANG 5.10: Delta table cho TAP 2 (Tatoeba/WMT)
# Source: Sheet Bang_5.9_Tap2_Tatoeba
# ============================================================
print()
print('=' * 65)
print('BANG 5.10 - Tap 2 (Tatoeba/WMT) - So sanh voi Clean')
print('=' * 65)

clean2    = {'acc': 71.27, 'f1': 71.27, 'bleu': 40.17}
fgsm2     = {'acc': 40.92, 'f1': 40.92, 'bleu': 12.44}
fgsm_def2 = {'acc': 63.74, 'f1': 63.74, 'bleu': 30.44}
cw2       = {'acc': 32.14, 'f1': 32.14, 'bleu':  6.40}
cw_def2   = {'acc': 57.56, 'f1': 57.56, 'bleu': 22.04}

rows2 = [
    ('FGSM (Chua phong thu)',   fgsm2,     None),
    ('FGSM + Co phong thu',     fgsm_def2, fgsm2),
    ('C+W (Chua phong thu)',    cw2,       None),
    ('C+W + Co phong thu',      cw_def2,   cw2),
]

for name, row, att in rows2:
    da = delta(row['acc'],  clean2['acc'])
    df = delta(row['f1'],   clean2['f1'])
    db = delta(row['bleu'], clean2['bleu'])
    if att is not None:
        rec = recovery(att['acc'], row['acc'], clean2['acc'])
        print(f'\n  [{name}]')
        print(f'    DAcc  = {row["acc"]}% - {clean2["acc"]}% = {da}%')
        print(f'    DF1   = {row["f1"]}% - {clean2["f1"]}% = {df}%')
        print(f'    DBLEU = {row["bleu"]}% - {clean2["bleu"]}% = {db}%')
        print(f'    PhucHoi = ({row["acc"]} - {att["acc"]}) / ({clean2["acc"]} - {att["acc"]}) x 100 = {rec}%')
    else:
        print(f'\n  [{name}]')
        print(f'    DAcc  = {row["acc"]}% - {clean2["acc"]}% = {da}%')
        print(f'    DF1   = {row["f1"]}% - {clean2["f1"]}% = {df}%')
        print(f'    DBLEU = {row["bleu"]}% - {clean2["bleu"]}% = {db}%')
        print(f'    PhucHoi = — (chua phong thu, khong ap dung)')

print()
print('=== KET QUA BANG 5.10 ===')
print(f'{"Kich ban":<28} {"DAcc":>9} {"DF1":>9} {"DBLEU":>9} {"PhucHoi":>12}')
print('-' * 70)
for name, row, att in rows2:
    da = delta(row['acc'],  clean2['acc'])
    df = delta(row['f1'],   clean2['f1'])
    db = delta(row['bleu'], clean2['bleu'])
    rec_s = f'{recovery(att["acc"], row["acc"], clean2["acc"])}%' if att else '—'
    print(f'{name:<28} {str(da)+"%":>9} {str(df)+"%":>9} {str(db)+"%":>9} {rec_s:>12}')
