import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

# 设置全局字体为 Times new roman，符合 Nature/Science 标准
plt.rcParams['font.family'] = 'Times new roman'
plt.rcParams['font.size'] = 12  # 减小字体大小以适应期刊要求
plt.rcParams['axes.linewidth'] = 0.8  # 细化轴线
plt.rcParams['axes.spines.top'] = False  # 移除顶部边框
plt.rcParams['axes.spines.right'] = False  # 移除右侧边框
plt.rcParams['xtick.major.size'] = 4  # 调整刻度大小
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# 函数：在箱线图上添加显著性检验标注
def add_significance_annotations(ax, data, platforms, y_max, y_offset, indicator):
    pairs = [('Fusion', 'BLS'), ('Fusion', 'ULS'), ('BLS', 'ULS')]
    for i, (plat1, plat2) in enumerate(pairs):
        group1 = data[(data['Indicator'] == indicator) & (data['Platform'] == plat1)]['Value']
        group2 = data[(data['Indicator'] == indicator) & (data['Platform'] == plat2)]['Value']
        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
        if p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            continue
        x1, x2 = platforms.index(plat1), platforms.index(plat2)
        y = y_max + y_offset * (i + 0.5)  # 调整 y_offset 比例
        ax.plot([x1 + 0.05, x2 - 0.05], [y, y], 'k-', lw=0.8)  # 细化线条
        ax.text((x1 + x2) / 2, y, sig, ha='center', va='bottom', fontsize=10, family='Times new roman')

# 函数：在百分偏差图上添加显著性检验标注
def add_dev_significance_annotations(ax, data, indicators, y_max, y_offset):
    for indicator in indicators:
        group1 = data[(data['Indicator'] == indicator) & (data['Deviation'] == 'B-F%')]['Percentage']
        group2 = data[(data['Indicator'] == indicator) & (data['Deviation'] == 'U-F%')]['Percentage']
        stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
        if p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            continue
        x1, x2 = indicators.index(indicator) - 0.15, indicators.index(indicator) + 0.15  # 缩小间距
        y = y_max + y_offset * ((indicators.index(indicator) % 3) + 0.5)
        ax.plot([x1, x2], [y, y], 'k-', lw=0.8)
        ax.text((x1 + x2) / 2, y, sig, ha='center', va='bottom', fontsize=10, family='Times new roman')

# 数据（与原代码一致）
gf_data = {
    'Fusion': [0.334341, 0.276261, 0.283533, 0.318523, 0.261804, 0.253143, 0.237947, 0.227104, 0.136377, 0.18382],
    'BLS': [0.480707, 0.426717, 0.430802, 0.472094, 0.418043, 0.567958, 0.370091, 0.369953, 0.1701, 0.345631],
    'ULS': [0.237474, 0.163017, 0.158674, 0.24753, 0.161441, 0.115682, 0.176447, 0.181372, 0.197514, 0.138083],
    'B-F%': [43.77746074, 54.4615418, 51.94069121, 48.21347281, 59.67785061, 124.3625145, 55.53505613, 62.90025715, 24.72777668, 88.02687412],
    'U-F%': [-28.97251608, -40.99167092, -44.03684933, -22.28818641, -38.33516677, -54.30171879, -25.84609178, -20.13702973, 44.82940672, -24.88140572]
}

ce_data = {
    'Fusion': [9.824988167, 9.934609735, 9.972630321, 9.912127699, 10.14264003, 10.44368188, 10.2110166, 10.04197599, 10.45958731, 10.46654372],
    'BLS': [9.578462718, 9.786417988, 9.744956026, 9.774891384, 9.957999137, 10.26247312, 10.16622158, 10.02639829, 10.42054783, 10.33863831],
    'ULS': [9.899315321, 10.00536582, 10.02902442, 10.02501153, 10.16406655, 10.02091512, 9.881092621, 9.90912739, 9.705556411, 10.37348666],
    'B-F%': [-2.509167897, -1.49167155, -2.282991424, -1.384529328, -1.820442138, -1.735104172, -0.438693048, -0.155125844, -0.373241112, -1.222040565],
    'U-F%': [0.756511384, 0.712218063, 0.565488714, 1.138845608, 0.211251902, -4.048062406, -3.23105908, -1.322932858, -7.208992828, -0.889090635]
}

pai_data = {
    'Fusion': [6.315698686, 6.981939505, 6.757994714, 6.095712215, 6.616474858, 6.073262662, 5.813357589, 5.711017501, 8.850075279, 5.599651669],
    'BLS': [6.253287973, 6.873856753, 6.60304728, 6.041669661, 6.437476332, 2.723516274, 5.789202127, 5.658542341, 9.10272794, 5.382755031],
    'ULS': [7.04582401, 8.092494377, 8.285314261, 6.608780543, 8.361634444, 8.695810345, 7.38202822, 7.143672911, 6.668872097, 8.43107722],
    'B-F%': [-0.988183827, -1.54803335, -2.292801941, -0.88656669, -2.70534582, -55.15563173, -0.415516535, -0.918840819, 2.854808044, -3.873395183],
    'U-F%': [11.56048381, 15.90610848, 22.60018854, 8.416872548, 26.37597245, 43.1818584, 26.98390056, 25.08581719, -24.646154, 50.56431575]
}

lamean_data = {
    'Fusion': [0.090882353, 0.094635105, 0.110962141, 0.088604367, 0.116058321, 0.117883334, 0.094262599, 0.087489434, 0.1434375, 0.113484815, 0.114535714, 0.130013741],
    'BLS': [0.184384709, 0.08234783, 0.071893949, 0.071853631, 0.100151709, 0.061374051, 0.097560163, 0.088133545, 0.117317699, 0.107214198, 0.109978736, 0.057517652],
    'ULS': [0.167034211, 0.167742196, 0.239344695, 0.142071972, 0.232686546, 0.130511259, 0.091472511, 0.074010169, 0.085616404, 0.172367443, 0.144329922, 0.165150247],
    'B-F%': [102.8828514, -12.98384484, -35.20857787, -18.90509087, -13.7057052, -47.93661745, 3.498273819, 0.736215549, -18.20988339, -5.52551199, -3.978652492, -55.76032818],
    'U-F%': [83.79168858, 77.25155588, 115.6994214, 60.34421022, 100.4910505, 10.71222266, -2.959910026, -15.40673515, -40.31100398, 51.88590932, 26.01302846, 27.02522489]
}

lacv_data = {
    'Fusion': [0.619978376, 0.621147723, 0.66275035, 0.635179628, 0.49551575, 0.502494854, 0.492896796, 0.477804174, 0.642853479, 0.448446303, 0.472685867, 0.468380572],
    'BLS': [1.501235025, 1.237896524, 1.468387175, 1.287553371, 0.996320841, 0.823688888, 0.750135564, 0.778819969, 0.802138187, 0.651789939, 0.608310093, 0.847035119],
    'ULS': [0.727486116, 0.881641712, 0.817259678, 0.819207875, 0.683762053, 0.770587131, 0.439810619, 0.792877355, 1.216178751, 0.921624459, 0.609550368, 0.79283492],
    'B-F%': [142.1431269, 99.2918074, 121.5596228, 102.7069691, 101.0674413, 63.91986522, 52.18917432, 62.99982533, 24.77776238, 45.34403207, 28.6922532, 80.8433503],
    'U-F%': [17.34056291, 41.93752612, 23.31335289, 28.9726307, 37.98997357, 53.35224311, -10.77024179, 65.94190634, 89.18443939, 105.5150087, 28.95464202, 69.2715212]
}

# 转换为 DataFrame
gf_df = pd.DataFrame(gf_data).melt(value_vars=['Fusion', 'BLS', 'ULS'], var_name='Platform', value_name='Value')
gf_df['Indicator'] = 'GF'

ce_df = pd.DataFrame(ce_data).melt(value_vars=['Fusion', 'BLS', 'ULS'], var_name='Platform', value_name='Value')
ce_df['Indicator'] = 'CE'

pai_df = pd.DataFrame(pai_data).melt(value_vars=['Fusion', 'BLS', 'ULS'], var_name='Platform', value_name='Value')
pai_df['Indicator'] = 'PAI'

la_mean_df = pd.DataFrame(lamean_data).melt(value_vars=['Fusion', 'BLS', 'ULS'], var_name='Platform', value_name='Value')
la_mean_df['Indicator'] = 'LAmean'

la_cv_df = pd.DataFrame(lacv_data).melt(value_vars=['Fusion', 'BLS', 'ULS'], var_name='Platform', value_name='Value')
la_cv_df['Indicator'] = 'LAcv'

# 百分偏差数据
gf_dev_df = pd.DataFrame(gf_data).melt(value_vars=['B-F%', 'U-F%'], var_name='Deviation', value_name='Percentage').reset_index(drop=True)
gf_dev_df['Indicator'] = 'GF'

ce_dev_df = pd.DataFrame(ce_data).melt(value_vars=['B-F%', 'U-F%'], var_name='Deviation', value_name='Percentage').reset_index(drop=True)
ce_dev_df['Indicator'] = 'CE'

pai_dev_df = pd.DataFrame(pai_data).melt(value_vars=['B-F%', 'U-F%'], var_name='Deviation', value_name='Percentage').reset_index(drop=True)
pai_dev_df['Indicator'] = 'PAI'

la_mean_dev_df = pd.DataFrame(lamean_data).melt(value_vars=['B-F%', 'U-F%'], var_name='Deviation', value_name='Percentage').reset_index(drop=True)
la_mean_dev_df['Indicator'] = 'LAmean'

la_cv_dev_df = pd.DataFrame(lacv_data).melt(value_vars=['B-F%', 'U-F%'], var_name='Deviation', value_name='Percentage').reset_index(drop=True)
la_cv_dev_df['Indicator'] = 'LAcv'

# 合并数据
all_data = pd.concat([gf_df, ce_df, pai_df, la_mean_df, la_cv_df], ignore_index=True)
all_dev_data = pd.concat([gf_dev_df, ce_dev_df, pai_dev_df, la_mean_dev_df, la_cv_dev_df], ignore_index=True)

# 创建图形和子图 (2行3列，宽度18cm，约7.1英寸)
fig, axes = plt.subplots(2, 3, figsize=(9.1, 6), dpi=600)

# 使用 colorblind 调色板
sns.set_palette("colorblind")

# 自定义 X 轴标题
x_axis_titles = ['', '', '', '', '', '']

# 绘制 GF 箱线图
sns.boxplot(x='Platform', y='Value', data=all_data[all_data['Indicator'] == 'GF'], ax=axes[0, 0], width=0.3, linewidth=0.8)
axes[0, 0].set_xlabel(x_axis_titles[0], fontsize=10, family='Times new roman')
axes[0, 0].set_ylabel('GF', fontsize=12, family='Times new roman')
axes[0, 0].text(-0.15, 1.05, 'a)', transform=axes[0, 0].transAxes, fontsize=18, family='Times new roman')
add_significance_annotations(axes[0, 0], all_data, ['Fusion', 'BLS', 'ULS'], y_max=max(all_data[all_data['Indicator'] == 'GF']['Value']), y_offset=0.05, indicator='GF')
axes[0, 0].set_ylim(-0.05, max(all_data[all_data['Indicator'] == 'GF']['Value']) + 0.15)  # 调整 Y 轴范围

# 绘制 CE 箱线图
sns.boxplot(x='Platform', y='Value', data=all_data[all_data['Indicator'] == 'CE'], ax=axes[0, 1], width=0.3, linewidth=0.8)
axes[0, 1].set_xlabel(x_axis_titles[1], fontsize=10, family='Times new roman')
axes[0, 1].set_ylabel('CE', fontsize=12, family='Times new roman')
axes[0, 1].text(-0.15, 1.05, 'b)', transform=axes[0, 1].transAxes, fontsize=18, family='Times new roman')
add_significance_annotations(axes[0, 1], all_data, ['Fusion', 'BLS', 'ULS'], y_max=max(all_data[all_data['Indicator'] == 'CE']['Value']), y_offset=0.3, indicator='CE')
axes[0, 1].set_ylim(min(all_data[all_data['Indicator'] == 'CE']['Value']) - 0.3, max(all_data[all_data['Indicator'] == 'CE']['Value']) + 0.9)

# 绘制 PAI 箱线图
sns.boxplot(x='Platform', y='Value', data=all_data[all_data['Indicator'] == 'PAI'], ax=axes[0, 2], width=0.3, linewidth=0.8)
axes[0, 2].set_xlabel(x_axis_titles[2], fontsize=10, family='Times new roman')
axes[0, 2].set_ylabel('PAI', fontsize=12, family='Times new roman')
axes[0, 2].text(-0.15, 1.05, 'c)', transform=axes[0, 2].transAxes, fontsize=18, family='Times new roman')
add_significance_annotations(axes[0, 2], all_data, ['Fusion', 'BLS', 'ULS'], y_max=max(all_data[all_data['Indicator'] == 'PAI']['Value']), y_offset=0.5, indicator='PAI')
axes[0, 2].set_ylim(min(all_data[all_data['Indicator'] == 'PAI']['Value']) - 0.5, max(all_data[all_data['Indicator'] == 'PAI']['Value']) + 1.5)

# 绘制 LAmean 箱线图
sns.boxplot(x='Platform', y='Value', data=all_data[all_data['Indicator'] == 'LAmean'], ax=axes[1, 0], width=0.3, linewidth=0.8)
axes[1, 0].set_xlabel(x_axis_titles[3], fontsize=10, family='Times new roman')
axes[1, 0].set_ylabel('LAmean', fontsize=12, family='Times new roman')
axes[1, 0].text(-0.15, 1.05, 'd)', transform=axes[1, 0].transAxes, fontsize=18, family='Times new roman')
add_significance_annotations(axes[1, 0], all_data, ['Fusion', 'BLS', 'ULS'], y_max=max(all_data[all_data['Indicator'] == 'LAmean']['Value']), y_offset=0.02, indicator='LAmean')
axes[1, 0].set_ylim(-0.02, max(all_data[all_data['Indicator'] == 'LAmean']['Value']) + 0.06)

# 绘制 LAcv 箱线图
sns.boxplot(x='Platform', y='Value', data=all_data[all_data['Indicator'] == 'LAcv'], ax=axes[1, 1], width=0.3, linewidth=0.8)
axes[1, 1].set_xlabel(x_axis_titles[4], fontsize=10, family='Times new roman')
axes[1, 1].set_ylabel('LAcv', fontsize=12, family='Times new roman')
axes[1, 1].text(-0.15, 1.05, 'e)', transform=axes[1, 1].transAxes, fontsize=18, family='Times new roman')
add_significance_annotations(axes[1, 1], all_data, ['Fusion', 'BLS', 'ULS'], y_max=max(all_data[all_data['Indicator'] == 'LAcv']['Value']), y_offset=0.1, indicator='LAcv')
axes[1, 1].set_ylim(-0.1, max(all_data[all_data['Indicator'] == 'LAcv']['Value']) + 0.3)

# 绘制百分偏差箱线图
sns.boxplot(x='Indicator', y='Percentage', hue='Deviation', data=all_dev_data, ax=axes[1, 2], width=0.4, linewidth=0.8)
axes[1, 2].set_xlabel(x_axis_titles[5], fontsize=10, family='Times new roman')
axes[1, 2].set_ylabel('Percentage Deviation (%)', fontsize=12, family='Times new roman')
axes[1, 2].text(-0.15, 1.05, 'f)', transform=axes[1, 2].transAxes, fontsize=18, family='Times new roman')
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=45, ha='right', fontsize=9, family='Times New Roman')  # 旋转 X 轴标签
axes[1, 2].legend(title='', fontsize=10, title_fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1))
# add_dev_significance_annotations(axes[1, 2], all_dev_data, ['GF', 'CE', 'PAI', 'LAmean', 'LAcv'], y_max=max(all_dev_data['Percentage']), y_offset=20)
axes[1, 2].set_ylim(min(all_dev_data['Percentage']) - 10, max(all_dev_data['Percentage']) + 60)

# 调整布局，优化间距
plt.tight_layout(pad=1.5)
plt.savefig('figure.png', dpi=600, bbox_inches='tight')  # 保存高分辨率图像
plt.show()