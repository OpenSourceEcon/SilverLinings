# Code to generate plot

# Import packages
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ogcore.demographics as demog
import ogcore.parameter_plots as pp
import ogcore.utils as utils

# Create total US births and deaths: 1950-2100 plot

# Read in birth and death data. Historical US births and deaths historical data
# and forecasts come from Our World in Data, "Births and deaths per year,
# United States: historic estimates with future projectsion based on the UN
# medium scenario",
# https://ourworldindata.org/grapher/births-and-deaths-projected-to-2100?country=~USA
# and "Birth rate vs. death rate, 2023",
# https://ourworldindata.org/grapher/birth-rate-vs-death-rate?country=~USA. The
# original source data come from "World Population Prospects", United Nations,
# https://population.un.org/wpp/.
birth_death_df = pd.read_csv(
    './data/owd_births_deaths.csv', header=9,
    dtype={
        'year':np.int32, 'births_per_1k_hst': np.float64,
        'births_per_1k_frc': np.float64, 'deaths_per_1k_hst': np.float64,
        'deaths_per_1k_frc': np.float64, 'tot_pop':np.float64,
        'tot_deaths_hst': np.float64, 'tot_deaths_frc': np.float64,
        'tot_births_hst': np.float64, 'tot_births_frc': np.float64,
    }
)
# birth_death_df

birth_death_tot_hst_df = birth_death_df[
    ['year', 'tot_births_hst', 'tot_deaths_hst']
][birth_death_df['year'] <= 2023]
years_hst = birth_death_tot_hst_df['year'].to_numpy()
tot_births_hst = birth_death_tot_hst_df['tot_births_hst'].to_numpy() / 1e6
tot_deaths_hst = birth_death_tot_hst_df['tot_deaths_hst'].to_numpy() / 1e6

birth_death_tot_frc_df = birth_death_df[
    ['year', 'tot_births_frc', 'tot_deaths_frc']
][birth_death_df['year'] > 2023]
years_frc = birth_death_tot_frc_df['year'].to_numpy()
tot_births_frc = birth_death_tot_frc_df['tot_births_frc'].to_numpy() / 1e6
tot_deaths_frc = birth_death_tot_frc_df['tot_deaths_frc'].to_numpy() / 1e6

fig1, ax1 = plt.subplots()
ax1.plot(
    years_hst, tot_births_hst, linestyle='-', color='blue', linewidth=3,
    label='Births (historical data)'
)
ax1.plot(
    years_frc, tot_births_frc, linestyle=':', color='blue', linewidth=3,
    label='Births (UN forecast)'
)
ax1.plot(
    years_hst, tot_deaths_hst, linestyle='-', color='red', linewidth=3,
    label='Deaths (historical data)'
)
ax1.plot(
    years_frc, tot_deaths_frc, linestyle=':', color='red', linewidth=3,
    label='Deaths (UN forecast)'
)
ax1.vlines(
    x=2023.5, ymin=1.0, ymax=5.0, color='black', linestyle='--', linewidth=1
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
plt.ylim(1.25, 4.75)
plt.xlabel("Year")
plt.ylabel("Total births or deaths (millions)")
plt.legend()
plt.title("Total US births and deaths by year, 1950-2100")
plt.savefig('./images/us_birth_death_tot.png')
plt.show()
plt.close()


# Create plot of US birth rates and death rates: 1950-2100

birth_death_rate_hst_df = birth_death_df[
    ['year', 'births_per_1k_hst', 'deaths_per_1k_hst']
][birth_death_df['year'] <= 2023]
years_hst = birth_death_rate_hst_df['year'].to_numpy()
birth_rates_hst = birth_death_rate_hst_df['births_per_1k_hst'].to_numpy()
death_rates_hst = birth_death_rate_hst_df['deaths_per_1k_hst'].to_numpy()

birth_death_rate_frc_df = birth_death_df[
    ['year', 'births_per_1k_frc', 'deaths_per_1k_frc']
][birth_death_df['year'] >= 2023]
years_frc = birth_death_rate_frc_df['year'].to_numpy()
birth_death_rate_frc_df
birth_rates_frc = birth_death_rate_frc_df['births_per_1k_frc'].to_numpy()
death_rates_frc = birth_death_rate_frc_df['deaths_per_1k_frc'].to_numpy()

fig2, ax2 = plt.subplots()
ax2.plot(
    years_hst, birth_rates_hst, linestyle='-', color='blue', linewidth=3,
    label='Birth rate (historical data)'
)
ax2.plot(
    years_frc, birth_rates_frc, linestyle=':', color='blue', linewidth=3,
    label='Birth rate (US Census forecast)'
)
ax2.plot(
    years_hst, death_rates_hst, linestyle='-', color='red', linewidth=3,
    label='Death rate (historical data)'
)
ax2.plot(
    years_frc, death_rates_frc, linestyle=':', color='red', linewidth=3,
    label='Death rate (US Census forecast)'
)
ax2.vlines(
    x=2023, ymin=5, ymax=30, color='black', linestyle='--', linewidth=1
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
plt.ylim(7.0, 26.0)
plt.xlabel("Year")
plt.ylabel("Births or deaths per 1,000 population")
plt.legend()
plt.title("US births and deaths per 1,000 pop, 1950-2100")
plt.savefig('./images/us_birth_death_rate.png')
plt.show()
plt.close()


# Create plot of  US birth rates and death rates: 1950-2023

fig2b, ax2b = plt.subplots()
ax2b.plot(
    years_hst, birth_rates_hst, linestyle='-', color='blue', linewidth=3,
    label='Birth rate'
)
ax2b.plot(
    years_hst, death_rates_hst, linestyle='-', color='red', linewidth=3,
    label='Death rate'
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
# plt.ylim(7.0, 26.0)
plt.xlabel("Year")
plt.ylabel("Births or deaths per 1,000 population")
plt.legend()
plt.title("US births and deaths per 1,000 pop, 1950-2023")
plt.savefig('./images/us_birth_death_rate2023.png')
plt.show()
plt.close()


# Create plot of US productivity growth rates

# Read in birth and death data. Historical measures of total factor
# productivity annual growth rates from the private nonfarm business sector
# from 1988 to 2023 come from FRED "Private Nonfarm Business Sector: Total
# Factor Productivity (MPU4910013)",
# https://fred.stlouisfed.org/series/MPU4910013. Historical measures of labor
# productivity annual growth rates from the private nonfarm business sector
# from 1988 to 2023 come from FRED "Private Nonfarm Business Sector: Labor
# Productivity (MPU4910063)", https://fred.stlouisfed.org/series/MPU4910063

prod_df = pd.read_csv(
    './data/us_productivity.csv', header=8,
    dtype={
        'year':np.int32, 'tfp_cnst_price_index': np.float64,
        'tfp_cnst_price_pctchg': np.float64,
		'tfp_prv_nf_bus_pctchg': np.float64,
        'tfp_prv_nf_bus_index': np.float64,
		'labprod_prv_nf_bus_pctchg':np.float64,
        'labprod_prv_nf_bus_index': np.float64
    }
)
# prod_df

prod_prv_nf_bus_df = prod_df[
    [
        'year', 'tfp_prv_nf_bus_pctchg', 'tfp_prv_nf_bus_index',
        'labprod_prv_nf_bus_pctchg', 'labprod_prv_nf_bus_index'
    ]
][prod_df['year'] >= 1988]
years = prod_prv_nf_bus_df['year'].to_numpy()
tfp_pctchg = prod_prv_nf_bus_df['tfp_prv_nf_bus_pctchg'].to_numpy()
labprod_pctchg = prod_prv_nf_bus_df['labprod_prv_nf_bus_pctchg'].to_numpy()

fig3, ax3= plt.subplots()
ax3.plot(
    years, tfp_pctchg, linestyle='-', color='blue', linewidth=2,
    label='Total factor productivity'
)
ax3.plot(
    years, labprod_pctchg, linestyle='-', color='green', linewidth=2,
    label='Labor productivity'
)
ax3.hlines(
    y=0.0, xmin=1987, xmax=2024, color='black', linestyle='--', linewidth=1
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
plt.xlim(1987, 2024)
plt.xlabel("Year")
plt.ylabel("Percent change")
plt.legend()
plt.title("US productivity annual growth rates: 1988-2023")
plt.savefig('./images/us_productivity_pctchg.png')
plt.show()
plt.close()

tfp_pctchg_mean2006_2023 = np.mean(
    prod_prv_nf_bus_df['tfp_prv_nf_bus_pctchg'][
        prod_prv_nf_bus_df['year'] >= 2006
    ].to_numpy()
)
labprod_pctchg_mean2006_2023 = np.mean(
    prod_prv_nf_bus_df['labprod_prv_nf_bus_pctchg'][
        prod_prv_nf_bus_df['year'] >= 2006
    ].to_numpy()
)
tfp_pctchg_mean1988_2005 = np.mean(
    prod_prv_nf_bus_df['tfp_prv_nf_bus_pctchg'][
        prod_prv_nf_bus_df['year'] < 2006
    ].to_numpy()
)
labprod_pctchg_mean1988_2005 = np.mean(
    prod_prv_nf_bus_df['labprod_prv_nf_bus_pctchg'][
        prod_prv_nf_bus_df['year'] < 2006
    ].to_numpy()
)
tfp_pctchg_mean1988_2023 = np.mean(tfp_pctchg)
labprod_pctchg_mean1988_2023 = np.mean(labprod_pctchg)
print(
    "Average labor productivity annual growth rate, " +
    "1988-2005: {:.2f}%".format(labprod_pctchg_mean1988_2005)
)
print("Average TFP annual growth rate, 1988-2005: {:.2f}%".format(
    tfp_pctchg_mean1988_2005
))
print(
    "Average labor productivity annual growth rate, " +
    "2006-2023: {:.2f}%".format(labprod_pctchg_mean2006_2023)
)
print("Average TFP annual growth rate, 2006-2023: {:.2f}%".format(
    tfp_pctchg_mean2006_2023
))
print(
    "Average labor productivity annual growth rate, " +
    "1988-2023: {:.2f}%".format(labprod_pctchg_mean1988_2023)
)
print("Average TFP annual growth rate, 1988-2023: {:.2f}%".format(
    tfp_pctchg_mean1988_2023
))


# Create plot of US productivity index

# I think this index is incorrect. I think you need to price adjust it, which
# the constant prices indices data through 2019 do.

tfp_index = prod_prv_nf_bus_df['tfp_prv_nf_bus_index'].to_numpy()
labprod_index = prod_prv_nf_bus_df['labprod_prv_nf_bus_index'].to_numpy()

fig4, ax4= plt.subplots()
ax4.plot(
    years, tfp_index, linestyle='-', color='blue', linewidth=2,
    label='Total factor productivity'
)
ax4.plot(
    years, labprod_index, linestyle='-', color='green', linewidth=2,
    label='Labor productivity'
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
# plt.xlim(1987, 2024)
plt.xlabel("Year")
plt.ylabel("Index (1988=1.0)")
plt.legend()
plt.title("US productivity index, 1988=1.0: 1988-2023")
plt.savefig('./images/us_productivity_index.png')
plt.show()
plt.close()


# Create plots of effects of 1gen, 2gen, drug effect on population over time:
# 2025-2050 (26 years)

cur_dir = os.path.dirname(os.path.realpath(__file__))
ogsims_dir = os.path.join(cur_dir, "data", "og_simulations")
years = np.arange(2025, 2051)
T = len(years)
# Read in baseline population time path from model output
(
    fert_rates_base_TP,
    mort_rates_base_TP,
    infmort_rates_base_TP,
    imm_rates_base_TP,
    pop_dist_base_TP,
    pre_pop_dist_base
) = utils.safe_read_pickle(
    os.path.join(ogsims_dir, "baseline", "demog_vars_baseline.pkl")
)

tot_pop_2025 = pop_dist_base_TP[0, :].sum()

tot_pop_2025_2050_base = np.zeros(T)
tot_pop_2025_2050_base[0] = tot_pop_2025

tot_pop_2025_2050_1gen = np.zeros(T)
tot_pop_2025_2050_1gen[0] = tot_pop_2025

tot_pop_2025_2050_2gen = np.zeros(T)
tot_pop_2025_2050_2gen[0] = tot_pop_2025

p_base = utils.safe_read_pickle(
    os.path.join(ogsims_dir, "baseline", "model_params.pkl")
)
g_n_vec_base = p_base.g_n
p_1gen = utils.safe_read_pickle(
    os.path.join(ogsims_dir, "section1_1gen", "model_params.pkl")
)
g_n_vec_1gen = p_1gen.g_n
p_2gen = utils.safe_read_pickle(
    os.path.join(ogsims_dir, "section1_2gen", "model_params.pkl")
)
g_n_vec_2gen = p_2gen.g_n
for t in range(1, T):
    tot_pop_2025_2050_base[t] = (
        (1 + g_n_vec_base[t - 1]) * tot_pop_2025_2050_base[t - 1]
    )
    tot_pop_2025_2050_1gen[t] = (
        (1 + g_n_vec_1gen[t - 1]) * tot_pop_2025_2050_1gen[t - 1]
    )
    tot_pop_2025_2050_2gen[t] = (
        (1 + g_n_vec_2gen[t - 1]) * tot_pop_2025_2050_2gen[t - 1]
    )

# Put time series in millions of people
tot_pop_2025_2050_base = tot_pop_2025_2050_base / 1e6
tot_pop_2025_2050_1gen = tot_pop_2025_2050_1gen / 1e6
tot_pop_2025_2050_2gen = tot_pop_2025_2050_2gen / 1e6

# Plot the population difference between the baseline total population and the
# 1st gen reform
fig5, ax5= plt.subplots()
ax5.plot(
    years, tot_pop_2025_2050_base, linestyle='-', color='black', linewidth=2,
    label='baseline'
)
ax5.plot(
    years, tot_pop_2025_2050_1gen, linestyle='-', color='blue', linewidth=2,
    label='1st gen'
)
ax5.plot(
    years, tot_pop_2025_2050_2gen, linestyle='-', color='green', linewidth=2,
    label='2nd gen'
)
ax5.vlines(
    x=2035, ymin=340, ymax=385, color='black', linestyle='--',
    label="begin effective year"
)
ax5.vlines(
    x=2045, ymin=340, ymax=385, color='black', linestyle=':',
    label="full effective year"
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
plt.ylim(340, 385)
plt.xlabel("Year")
plt.ylabel("Population (millions)")
plt.legend()
plt.title("US population, 3 scenarios: 2025-2050")
plt.savefig('./images/us_pop_1st2ndgen.png')
plt.show()
plt.close()

# Plot the population difference between the baseline total population and the
# 1st gen reform and between the 2nd gen and 1st gen reform
fig6, ax6= plt.subplots()
ax6.plot(
    years, tot_pop_2025_2050_1gen - tot_pop_2025_2050_base, linestyle='-',
    color='blue', linewidth=3, label='1st gen - baseline'
)
ax6.plot(
    years, tot_pop_2025_2050_2gen - tot_pop_2025_2050_1gen, linestyle='-',
    color='green', linewidth=3, label='2nd gen - 1st gen'
)
ax6.vlines(
    x=2035, ymin=-0.05, ymax=1.2, color='black', linestyle='--',
    label="begin effective year"
)
ax6.vlines(
    x=2045, ymin=-0.05, ymax=1.2, color='black', linestyle=':',
    label="full effective year"
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
plt.ylim(-0.20, 1.2)
plt.xlabel("Year")
plt.ylabel("Population difference (millions)")
plt.legend()
plt.title("US population difference by year: 2025-2050")
plt.savefig('./images/us_popdiff_2nd1stgen.png')
plt.show()
plt.close()

diff_2gen_1gen = tot_pop_2025_2050_2gen - tot_pop_2025_2050_1gen
diff_2gen_base = tot_pop_2025_2050_2gen - tot_pop_2025_2050_base
print(diff_2gen_1gen)
print(diff_2gen_base)
print(np.arange(2025, 2051))


# Create plots of effects of 1-year and 2.5-year effects on population over
# time: 2025-2100 (76 years)

years = np.arange(2025, 2101)
T = len(years)

tot_pop_2025_2100_base = np.zeros(T)
tot_pop_2025_2100_base[0] = tot_pop_2025

tot_pop_2025_2100_1yr_all = np.zeros(T)
tot_pop_2025_2100_1yr_all[0] = tot_pop_2025

tot_pop_2025_2100_2p5yr_all = np.zeros(T)
tot_pop_2025_2100_2p5yr_all[0] = tot_pop_2025

p_1yr_all = utils.safe_read_pickle(
    os.path.join(ogsims_dir, "section5_all", "model_params.pkl")
)
g_n_vec_1yr_all = p_1yr_all.g_n

p_2p5yr_all =utils.safe_read_pickle(
    os.path.join(ogsims_dir, "moonshot_section5_all", "model_params.pkl")
)
g_n_vec_2p5yr_all = p_2p5yr_all.g_n

for t in range(1, T):
    tot_pop_2025_2100_base[t] = (
        (1 + g_n_vec_base[t - 1]) * tot_pop_2025_2100_base[t - 1]
    )
    tot_pop_2025_2100_1yr_all[t] = (
        (1 + g_n_vec_1yr_all[t - 1]) * tot_pop_2025_2100_1yr_all[t - 1]
    )
    tot_pop_2025_2100_2p5yr_all[t] = (
        (1 + g_n_vec_2p5yr_all[t - 1]) * tot_pop_2025_2100_2p5yr_all[t - 1]
    )

# Put time series in millions of people
tot_pop_2025_2100_base = tot_pop_2025_2100_base / 1e6
tot_pop_2025_2100_1yr_all = tot_pop_2025_2100_1yr_all / 1e6
tot_pop_2025_2100_2p5yr_all = tot_pop_2025_2100_2p5yr_all / 1e6

# Plot the population difference between the baseline total population and the
# 1 year improvement reform
fig7, ax7= plt.subplots()
ax7.plot(
    years, tot_pop_2025_2100_base, linestyle='-', color='black', linewidth=2,
    label='baseline'
)
ax7.plot(
    years, tot_pop_2025_2100_1yr_all, linestyle='-', color='blue', linewidth=2,
    label='1yr improvement'
)
ax7.plot(
    years, tot_pop_2025_2100_2p5yr_all, linestyle='-', color='red',
    linewidth=2, label='2.5 yr improvement'
)
ax7.vlines(
    x=2035, ymin=340, ymax=400, color='black', linestyle='--',
    label="begin effective year"
)
ax7.vlines(
    x=2045, ymin=340, ymax=400, color='black', linestyle=':',
    label="full effective year"
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
plt.ylim(340, 400)
plt.xlabel("Year")
plt.ylabel("Population (millions)")
plt.legend()
plt.title("US population, 1-year and 2.5-year improvements: 2025-2100")
plt.savefig('./images/us_pop_1yr2p5yr_all.png')
plt.show()
plt.close()

# Plot the population difference between the baseline total population and the
# 1 yr improvement (all) reform and baseline and the 2.5-yr improvement (all)
fig8, ax8= plt.subplots()
ax8.plot(
    years, tot_pop_2025_2100_1yr_all - tot_pop_2025_2100_base, linestyle='-',
    color='blue', linewidth=3, label='1yr improv - baseline'
)
ax8.plot(
    years, tot_pop_2025_2100_2p5yr_all - tot_pop_2025_2100_base, linestyle='-',
    color='red', linewidth=3, label='2.5yr improv - baseline'
)
ax8.vlines(
    x=2035, ymin=-0.4, ymax=20.5, color='black', linestyle='--',
    label="begin effective year"
)
ax8.vlines(
    x=2045, ymin=-0.4, ymax=20.5, color='black', linestyle=':',
    label="full effective year"
)
plt.grid(
    visible=True, which='major', axis='both', color='0.5', linestyle='--',
    linewidth=0.3
)
plt.ylim(-0.4, 20.5)
plt.xlabel("Year")
plt.ylabel("Population difference (millions)")
# Put legend in lower right corner
plt.legend(loc='center right')
plt.title("US population difference by year: 2025-2100")
plt.savefig('./images/us_popdiff_1yr2p5yr_all.png')
plt.show()
plt.close()

diff_1yr_all_base = tot_pop_2025_2100_1yr_all - tot_pop_2025_2100_base
diff_2p5yr_all_base = tot_pop_2025_2100_2p5yr_all - tot_pop_2025_2100_base
print(diff_1yr_all_base)
print(diff_2p5yr_all_base)
print(np.arange(2025, 2101))

discount_rate = 0.04
discount_rate_vec = np.zeros_like(diff_1yr_all_base)
for t in range(len(discount_rate_vec)):
    discount_rate_vec[t] = (
        (1 + discount_rate) ** t
    )
print(discount_rate_vec)

qaly_npv_1yr_all = (
    (diff_1yr_all_base * 1e6 * 100_000) / discount_rate_vec
).sum()
qaly_npv_2p5yr_all = (
    (diff_2p5yr_all_base * 1e6 * 100_000) / discount_rate_vec
).sum()
print("QALY NPV of 1yr all reform is ($trillions):", qaly_npv_1yr_all / 1e12)
print(
    "QALY NPV of 2.5yr all reform is ($trillions):", qaly_npv_2p5yr_all / 1e12
)
