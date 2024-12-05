import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Diabeteszesek adatai
diab = pd.read_csv("diabetes.csv")
diab_both_sexes = diab[diab["Dim1ValueCode"] == "SEX_BTSX"]
diab_dict = diab_both_sexes.to_dict(orient="records")
diab_values = dict(zip(diab_both_sexes["Period"], diab_both_sexes["FactValueNumeric"]))
diab_values = dict(sorted(diab_values.items()))
'''
# Magas vérnyomás adatai
ht = pd.read_csv("hypertension.csv")
ht_both_sexes = ht[ht["Dim1ValueCode"] == "SEX_BTSX"]
ht_dict = ht_both_sexes.to_dict(orient="records")
ht_values = dict(zip(ht_both_sexes["Period"], ht_both_sexes["FactValueNumeric"]))
ht_values = dict(sorted(ht_values.items()))

# BMI adatai
bmi = pd.read_csv("bmi.csv")
bmi_both_sexes = bmi[bmi["Dim1ValueCode"] == "SEX_BTSX"]
bmi_dict = bmi_both_sexes.to_dict(orient="records")
bmi_values = dict(zip(bmi_both_sexes["Period"], bmi_both_sexes["FactValueNumeric"]))
bmi_values = dict(sorted(bmi_values.items()))
'''
# GDB per capita adatai
gdp = pd.read_csv("gdp.csv")
gdp = gdp[pd.to_datetime(gdp["DATE"]).dt.year >= 1990]
gdp = gdp[pd.to_datetime(gdp["DATE"]).dt.year < 2023]
gdp_dict = dict(zip(pd.to_datetime(gdp["DATE"]).dt.year,  gdp["PCAGDPRWA646NWDB"]))


years = list(diab_values.keys())

# Szűrés azután az év után ahol már gdp és diabetes adat is van
diab_array = list(diab_values.values())
mean_diab = np.mean(diab_array)
tapasztali_szoras_diab = np.std(diab_array, ddof=0)
korr_tapasztali_szoras_diab = np.std(diab_array, ddof=1)
print(f" A diabéteszesek prevalenciájának átlaga: {mean_diab:.4f} %")
print(f" A diabéteszesek prevalenciájának tapasztalati szórása: {tapasztali_szoras_diab:.4f} %")
print(f" A diabéteszesek prevalenciájának korrigált tapasztalati szórása: {korr_tapasztali_szoras_diab:.4f} %")

gdp_array = list(gdp_dict.values())
mean_gdp = np.mean(gdp_array)
tapasztali_szoras_gdp = np.std(gdp_array, ddof=0)
korr_tapasztali_szoras_gdp = np.std(gdp_array, ddof=1)
print(f" Az egy főre jutó GDP átlaga: {mean_gdp:.4f} %")
print(f" Az egy főre jutó GDP tapasztalati szórása: {tapasztali_szoras_gdp:.4f} %")
print(f" Az egy főre jutó GDP korrigált tapasztalati szórása: {korr_tapasztali_szoras_gdp:.4f} %")

# Boxplot diab
plt.boxplot(diab_array)
plt.title('Diabeteszesek előfordulása boxplot')
plt.ylabel('Előfordulás [%]')
plt.savefig("kepek/boxplot_diab.png", dpi=300, bbox_inches='tight')
plt.show()

# Tapasztalati sűrűségfüggvény diab
sns.histplot(diab_array, kde=False, stat="density", bins=10, color='blue', alpha=0.6)
plt.title('Diabéteszesek arányának tapasztalati sűrűségfüggvénye')
plt.xlabel('Előfordulás [%]')
plt.ylabel('Gyakoriság')
plt.show()

# Boxplot gdp
gdp_array = list(gdp_dict.values())
plt.boxplot(gdp_array)
plt.title('Egy főre jutó GDP boxplot')
plt.ylabel('Egy főre jutó GDP[$]')
plt.savefig("kepek/boxplot_gdp.png", dpi=300, bbox_inches='tight')
plt.show()

# Tapasztalati sűrűségfüggvény gdp
bins_array = [min(gdp_array), 300, 400, 600, 700, 850, max(gdp_array)]
sns.histplot(gdp_array, kde=False, stat="density", bins=bins_array, color='blue', alpha=0.6)
plt.title('Egy főre jutó GDP tapasztalati sűrűségfüggvénye')
plt.xlabel('GDP [$]')
plt.ylabel('Gyakoriság')
plt.savefig("kepek/hisztogram_gdp.png", dpi=300, bbox_inches='tight')
plt.show()

bins_array = [min(diab_array), 4.25, 4.5, 4.75, 5, 5.5, max(diab_array)]
sns.histplot(diab_array, kde=False, stat="density", bins=bins_array, color='blue', alpha=0.6)
plt.title('Diabéteszesek arányának tapasztalati sűrűségfüggvénye')
plt.xlabel('Előfordulás [%]')
plt.ylabel('Gyakoriság')
plt.savefig("kepek/hisztogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Korrelációk
'''
plt.figure(figsize=(10, 6))
plt.plot(years, diab_array)
plt.title('Diabéteszesek előforulása idővel')
plt.xlabel('Idő [év]')
plt.ylabel('Előfordulás [%]')
plt.savefig("kepek/diabeteszesek.png", dpi=300, bbox_inches='tight')
plt.show()

gdp_array = list(gdp_dict.values())
plt.figure(figsize=(10, 6))
plt.plot(years, gdp_array)
plt.title('Egy főre jutó GDP')
plt.xlabel('Idő [év]')
plt.ylabel('GDP per capita [$]')
plt.savefig("kepek/gdp.png", dpi=300, bbox_inches='tight')
plt.show()
'''
pearson = scipy.stats.pearsonr(diab_array, gdp_array)
spearman = scipy.stats.spearmanr(diab_array, gdp_array)
kendall = scipy.stats.kendalltau(diab_array, gdp_array)

print(f"Pearson korreláció: {pearson}")
print(f"Spearman korreláció: {spearman}")
print(f"Kendall korreláció: {kendall}")


# Függvény illesztés
r = int(len(gdp_array) / 2)
x1 = np.mean(gdp_array[:r])
x2 = np.mean(gdp_array[r:])
y1 = np.mean(diab_array[:r])
y2 = np.mean(diab_array[r:])
a = (y2-y1) / (x2-x1)
b = y1 - a*x1
def f(x, a, b):
    return a*x+b

x_line = np.linspace(min(gdp_array), max(gdp_array), 33)
y_line = f(x_line, a, b)
print(f"Illesztett függvény: {a}*x+{b}")

plt.scatter(gdp_array, np.array(diab_array))
plt.plot(x_line, y_line, "red")
plt.xlabel('Egy főre jutó GDP [$]')
plt.ylabel('Diabéteszesek előfordulása [%]')
plt.savefig("kepek/fuggvenyillesztes.png", dpi=300, bbox_inches='tight')
plt.show()

# Determinációs együttható
St2 = (1/ len(gdp_array)) * sum((diab_array - np.mean(diab_array))**2)
Sm2 = (1/ len(gdp_array)) * sum((diab_array - y_line)**2)
R2 =(St2 - Sm2) / St2
print(f"Determinációs együttható a fgv. illesztésre: {R2}")

# Abbe paraméteres próba
q2_diab = np.sum(np.diff(diab_array)**2)
q2_gdp = np.sum(np.diff(gdp_array)**2)
q2_diab /=  (2*(len(diab_array) - 1))
q2_gdp /= (2*(len(gdp_array) - 1))

rakt_diab = q2_diab / (tapasztali_szoras_diab**2)
rakt_gdp = q2_gdp / (korr_tapasztali_szoras_gdp**2)

# 33 elemünk van, ezért táblázatból:
rkrit = 0.722

print(f"\nA diabéteszesek tekintetében, mivel r_akt={rakt_diab} és r_krit={rkrit}, ezért a hipotézist elutasítom.")
print(f"A gdp tekintetében, mivel r_akt={rakt_gdp} és r_krit={rkrit}, ezért a hipotézist elutasítom.")
