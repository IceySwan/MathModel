# -*- coding: utf-8 -*-
"""
LED太阳光模拟优化系统
基于五通道LED合成光模拟对应时刻的太阳光
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import optimize, interpolate
import seaborn as sns
import warnings
import time
import platform

warnings.filterwarnings('ignore')


# 设置中文字体
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()

    if system == "Windows":
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif system == "Darwin":  # macOS
        font_candidates = ['PingFang SC', 'Arial Unicode MS', 'Helvetica']
    else:  # Linux
        font_candidates = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_font = None
    for font in font_candidates:
        if font in available_fonts:
            chinese_font = font
            break

    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['sans-serif']

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300


setup_chinese_font()
sns.set_style("whitegrid")


class LEDSunlightOptimizer:
    """LED太阳光模拟优化器"""

    def __init__(self):
        """初始化"""
        self.wavelengths = np.arange(380, 781, 1)  # 380-780nm, 1nm间隔
        self.setup_cie_functions()
        self.setup_planck_locus()
        self.setup_melanopic_function()

        # 优化权重参数
        self.alpha = {
            'cct': 0.01,  # α1 = 0.01
            'duv': 100,  # α2 = 10
            'rf': 0.1,  # α3 = 0.1
            'rg': 0.1,  # α4 = 0.1
            'der_mel': 1000  # α5 = 1
        }

        print(f"优化权重参数: {self.alpha}")

    def setup_cie_functions(self):
        """设置CIE标准函数"""
        wl = self.wavelengths

        # X颜色匹配函数
        x_main = 1.056 * np.exp(-0.5 * ((wl - 599.8) / 37.9) ** 2)
        x_secondary = 0.362 * np.exp(-0.5 * ((wl - 442.0) / 16.4) ** 2)
        x_negative = 0.065 * np.exp(-0.5 * ((wl - 501.1) / 20.9) ** 2)
        self.x_bar = x_main + x_secondary - x_negative

        # Y颜色匹配函数
        self.y_bar = np.exp(-0.5 * ((wl - 556.1) / 46.9) ** 2)

        # Z颜色匹配函数
        self.z_bar = 1.669 * np.exp(-0.5 * ((wl - 449.8) / 19.4) ** 2)

        # 确保非负并归一化
        self.x_bar = np.maximum(self.x_bar, 0)
        self.y_bar = np.maximum(self.y_bar, 0)
        self.z_bar = np.maximum(self.z_bar, 0)
        self.y_bar = self.y_bar / np.max(self.y_bar)

    def setup_planck_locus(self):
        """设置普朗克轨迹"""
        self.cct_range = np.arange(1000, 25001, 100)
        self.planck_x = []
        self.planck_y = []

        for cct in self.cct_range:
            spd_planck = self.planck_spd(cct)
            x, y, z = self.calculate_xyz(spd_planck)
            if x + y + z > 0:
                self.planck_x.append(x / (x + y + z))
                self.planck_y.append(y / (x + y + z))
            else:
                self.planck_x.append(0)
                self.planck_y.append(0)

        self.planck_x = np.array(self.planck_x)
        self.planck_y = np.array(self.planck_y)

    def setup_melanopic_function(self):
        """设置褪黑激素效应函数"""
        wl = self.wavelengths
        mel_function = np.zeros_like(wl)

        # 短波段（380-500nm）：主要响应区域
        short_wave_mask = (wl >= 380) & (wl <= 500)
        wl_short = wl[short_wave_mask]
        alpha = 0.5 * ((np.log(wl_short) - np.log(480)) / 0.15) ** 2
        mel_function[short_wave_mask] = np.exp(-alpha)

        # 长波段（500-780nm）：较弱的响应
        long_wave_mask = (wl > 500) & (wl <= 780)
        wl_long = wl[long_wave_mask]
        mel_function[long_wave_mask] = 0.1 * np.exp(-(wl_long - 500) / 100)

        self.mel_function = mel_function / np.max(mel_function)
        self.mel_function = np.maximum(self.mel_function, 0)

    def planck_spd(self, temperature):
        """计算普朗克黑体辐射SPD"""
        h = 6.626e-34
        c = 3e8
        k = 1.381e-23

        wl_m = self.wavelengths * 1e-9
        spd = (2 * h * c ** 2 / wl_m ** 5) / (np.exp(h * c / (wl_m * k * temperature)) - 1)
        spd = spd / np.max(spd) * 100
        return spd

    def calculate_xyz(self, spd):
        """计算XYZ三刺激值"""
        if len(spd) != len(self.wavelengths):
            f = interpolate.interp1d(np.linspace(380, 780, len(spd)), spd,
                                     bounds_error=False, fill_value=0)
            spd = f(self.wavelengths)

        X = np.trapz(spd * self.x_bar, self.wavelengths)
        Y = np.trapz(spd * self.y_bar, self.wavelengths)
        Z = np.trapz(spd * self.z_bar, self.wavelengths)
        return X, Y, Z

    def calculate_cct_duv(self, spd):
        """计算相关色温CCT和Duv"""
        X, Y, Z = self.calculate_xyz(spd)

        if X + Y + Z == 0:
            return 0, 0

        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)

        # 使用Robertson方法计算CCT
        distances = np.sqrt((self.planck_x - x) ** 2 + (self.planck_y - y) ** 2)
        min_idx = np.argmin(distances)

        if min_idx > 0 and min_idx < len(self.cct_range) - 1:
            idx_range = [min_idx - 1, min_idx, min_idx + 1]
            x_pts = self.planck_x[idx_range]
            y_pts = self.planck_y[idx_range]
            cct_pts = self.cct_range[idx_range]

            dists = [(x - x_pts[i]) ** 2 + (y - y_pts[i]) ** 2 for i in range(3)]
            weights = [1 / max(d, 1e-10) for d in dists]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            cct = sum(cct_pts[i] * weights[i] for i in range(3))
            closest_x = sum(x_pts[i] * weights[i] for i in range(3))
            closest_y = sum(y_pts[i] * weights[i] for i in range(3))
        else:
            cct = self.cct_range[min_idx]
            closest_x = self.planck_x[min_idx]
            closest_y = self.planck_y[min_idx]

        # Duv计算
        duv = np.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
        if y > closest_y:
            duv = duv
        else:
            duv = -duv

        return cct, duv

    def calculate_rf_rg(self, spd):
        """计算保真度指数Rf和色域指数Rg"""
        cct, _ = self.calculate_cct_duv(spd)

        if cct < 5000:
            ref_spd = self.planck_spd(cct)
        else:
            ref_spd = self.daylight_spd(cct)

        test_colors = self.get_test_color_reflectances()

        test_colors_under_test = []
        test_colors_under_ref = []

        for reflectance in test_colors:
            test_spectrum = spd * reflectance
            X_t, Y_t, Z_t = self.calculate_xyz(test_spectrum)
            if X_t + Y_t + Z_t > 0:
                test_colors_under_test.append([X_t / (X_t + Y_t + Z_t), Y_t / (X_t + Y_t + Z_t)])
            else:
                test_colors_under_test.append([0.33, 0.33])

            ref_spectrum = ref_spd * reflectance
            X_r, Y_r, Z_r = self.calculate_xyz(ref_spectrum)
            if X_r + Y_r + Z_r > 0:
                test_colors_under_ref.append([X_r / (X_r + Y_r + Z_r), Y_r / (X_r + Y_r + Z_r)])
            else:
                test_colors_under_ref.append([0.33, 0.33])

        color_differences = []
        for i in range(len(test_colors)):
            x_t, y_t = test_colors_under_test[i]
            x_r, y_r = test_colors_under_ref[i]
            delta_e = np.sqrt((x_t - x_r) ** 2 + (y_t - y_r) ** 2) * 100
            color_differences.append(delta_e)

        avg_color_diff = np.mean(color_differences)
        rf = max(0, 100 - avg_color_diff * 4.6)

        test_area = self.calculate_color_gamut_area(test_colors_under_test)
        ref_area = self.calculate_color_gamut_area(test_colors_under_ref)

        if ref_area > 0:
            rg = (test_area / ref_area) * 100
        else:
            rg = 100

        rg = max(50, min(150, rg))
        return rf, rg

    def get_test_color_reflectances(self):
        """获取测试颜色的反射光谱"""
        colors = []
        wl = self.wavelengths

        red = 0.1 + 0.8 * np.exp(-((wl - 700) / 50) ** 2)
        orange = 0.1 + 0.7 * np.exp(-((wl - 600) / 40) ** 2)
        yellow = 0.2 + 0.6 * np.exp(-((wl - 580) / 35) ** 2)
        green = 0.1 + 0.7 * np.exp(-((wl - 530) / 30) ** 2)
        cyan = 0.1 + 0.6 * np.exp(-((wl - 480) / 25) ** 2)
        blue = 0.1 + 0.8 * np.exp(-((wl - 450) / 30) ** 2)
        purple = 0.1 + 0.6 * np.exp(-((wl - 420) / 25) ** 2)
        skin = 0.3 + 0.4 * np.exp(-((wl - 600) / 80) ** 2)

        colors.extend([red, orange, yellow, green, cyan, blue, purple, skin])
        return colors

    def calculate_color_gamut_area(self, color_points):
        """计算色域面积"""
        if len(color_points) < 3:
            return 0

        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(color_points)
            return hull.volume
        except:
            return 0

    def daylight_spd(self, cct):
        """标准日光D系列SPD"""
        wl = self.wavelengths

        if cct <= 4000:
            spd = 100 * (1 + 0.3 * np.exp(-((wl - 650) / 100) ** 2))
        elif cct >= 8000:
            spd = 100 * (1 + 0.5 * np.exp(-((wl - 450) / 80) ** 2))
        else:
            spd = 100 * np.ones_like(wl)

        return spd

    def calculate_mel_der(self, spd):
        """计算褪黑激素日光效率比mel-DER"""
        mel_irradiance = np.trapz(spd * self.mel_function, self.wavelengths)
        photopic_irradiance = np.trapz(spd * self.y_bar, self.wavelengths)

        if photopic_irradiance == 0:
            return 0

        return mel_irradiance / photopic_irradiance

    def load_led_data(self):
        """加载LED数据"""
        try:
            # 读取LED数据
            led_data = pd.read_csv('attachment-Q2.csv')
            print(f"LED数据形状: {led_data.shape}")
            print("LED数据列名:", list(led_data.columns))

            # 处理列名（去除多余空格和引号）
            led_data.columns = [col.strip().strip('"') for col in led_data.columns]
            print("处理后的列名:", list(led_data.columns))

            # 提取波长和LED通道数据
            wavelength_col = led_data.columns[0]  # 第一列是波长
            led_channels = led_data.columns[1:6]  # 前5个LED通道

            wavelengths_raw = led_data[wavelength_col].values
            led_spds_raw = led_data[led_channels].values

            # 清理数据
            wavelengths = []
            led_spds_clean = []

            for i, wl in enumerate(wavelengths_raw):
                try:
                    if isinstance(wl, str):
                        wl_num = float(wl.split('(')[0]) if '(' in wl else float(wl)
                    else:
                        wl_num = float(wl)

                    led_row = []
                    valid_row = True
                    for j in range(min(5, len(led_channels))):
                        try:
                            val = float(led_spds_raw[i, j])
                            if np.isnan(val):
                                valid_row = False
                                break
                            led_row.append(val)
                        except:
                            valid_row = False
                            break

                    if valid_row and not np.isnan(wl_num) and len(led_row) == 5:
                        wavelengths.append(wl_num)
                        led_spds_clean.append(led_row)
                except:
                    continue

            wavelengths = np.array(wavelengths)
            led_spds = np.array(led_spds_clean)

            print(f"有效LED数据点数: {len(wavelengths)}")
            print(f"波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

            # 插值到标准波长网格
            led_spds_standard = np.zeros((len(self.wavelengths), 5))
            for i in range(5):
                f = interpolate.interp1d(wavelengths, led_spds[:, i],
                                         bounds_error=False, fill_value=0)
                led_spds_standard[:, i] = f(self.wavelengths)

            self.led_spds = led_spds_standard
            self.led_channel_names = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']

            print("LED光谱数据加载成功！")
            return True

        except Exception as e:
            print(f"加载LED数据时出错: {e}")
            print("使用模拟LED数据...")

            # 生成模拟LED数据
            led_spds_standard = np.zeros((len(self.wavelengths), 5))

            # 蓝光 (450nm)
            blue = 100 * np.exp(-0.5 * ((self.wavelengths - 450) / 20) ** 2)
            led_spds_standard[:, 0] = blue

            # 绿光 (530nm)
            green = 100 * np.exp(-0.5 * ((self.wavelengths - 530) / 25) ** 2)
            led_spds_standard[:, 1] = green

            # 红光 (660nm)
            red = 100 * np.exp(-0.5 * ((self.wavelengths - 660) / 20) ** 2)
            led_spds_standard[:, 2] = red

            # 暖白光 (3000K)
            ww_spd = self.planck_spd(3000)
            led_spds_standard[:, 3] = ww_spd

            # 冷白光 (6500K)
            cw_spd = self.planck_spd(6500)
            led_spds_standard[:, 4] = cw_spd

            self.led_spds = led_spds_standard
            self.led_channel_names = ['Blue', 'Green', 'Red', 'Warm White', 'Cold White']

            print("模拟LED数据生成成功！")
            return False

    def load_sunlight_data(self):
        """加载太阳光数据"""
        try:
            # 读取太阳光数据，使用quotechar参数处理引号
            sunlight_data = pd.read_csv('attachment-Q3.csv', quotechar='"')
            print(f"太阳光数据形状: {sunlight_data.shape}")
            print("太阳光数据列名:", list(sunlight_data.columns))

            # 处理列名（去除多余空格和引号）
            sunlight_data.columns = [col.strip().strip('"').strip() for col in sunlight_data.columns]
            print("处理后的列名:", list(sunlight_data.columns))

            # 重命名列以便于使用
            column_mapping = {
                'Time': 'Time',
                'CCT(K)': 'CCT',
                'Duv': 'Duv',
                'Rf': 'Rf',
                'Rg': 'Rg',
                'DER_mel': 'DER_mel'
            }

            # 找到实际的列名并重命名
            new_columns = {}
            for col in sunlight_data.columns:
                for old_name, new_name in column_mapping.items():
                    if old_name in col or col in old_name:
                        new_columns[col] = new_name
                        break

            sunlight_data = sunlight_data.rename(columns=new_columns)
            print("重命名后的列名:", list(sunlight_data.columns))

            # 处理数据中的引号并转换为数值类型
            for col in sunlight_data.columns:
                if col != 'Time':  # 除了时间列，其他都转换为数值
                    # 去除引号并转换为数值
                    sunlight_data[col] = sunlight_data[col].astype(str).str.strip().str.strip('"')
                    sunlight_data[col] = pd.to_numeric(sunlight_data[col], errors='coerce')
                else:
                    # 处理时间列的引号
                    sunlight_data[col] = sunlight_data[col].astype(str).str.strip().str.strip('"')

            self.sunlight_data = sunlight_data
            print("太阳光数据加载成功！")
            print("数据预览:")
            print(sunlight_data.head())

            # 检查是否有NaN值
            nan_count = sunlight_data.isnull().sum()
            if nan_count.any():
                print("警告：发现NaN值:")
                print(nan_count[nan_count > 0])
            else:
                print("所有数据均有效，无NaN值")

            return True

        except Exception as e:
            print(f"加载太阳光数据时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def objective_function(self, weights, target_params):
        """优化目标函数"""
        if np.sum(weights) == 0:
            return 1e10

        # 归一化权重
        weights = weights / np.sum(weights)

        # 合成LED光谱
        combined_spd = np.dot(self.led_spds, weights)

        # 计算LED光的参数
        cct_led, duv_led = self.calculate_cct_duv(combined_spd)
        rf_led, rg_led = self.calculate_rf_rg(combined_spd)
        der_mel_led =  2 * self.calculate_mel_der(combined_spd)

        # 计算偏差
        delta_cct = abs(cct_led - target_params['CCT'])
        delta_duv = abs(duv_led - target_params['Duv'])
        delta_rf = abs(rf_led - target_params['Rf'])
        delta_rg = abs(rg_led - target_params['Rg'])
        delta_der_mel = abs(der_mel_led - target_params['DER_mel'])

        # 计算目标函数值
        objective = (self.alpha['cct'] * delta_cct +
                     self.alpha['duv'] * delta_duv +
                     self.alpha['rf'] * delta_rf +
                     self.alpha['rg'] * delta_rg +
                     self.alpha['der_mel'] * delta_der_mel)

        return objective

    def optimize_single_time(self, target_params, max_time=30):
        """优化单个时刻的LED权重"""

        # 约束条件：权重和为1，权重非负
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(5)]

        # 多个初始猜测以提高优化效果
        initial_guesses = [
            np.ones(5) / 5,  # 均匀分布
            np.array([0.1, 0.2, 0.1, 0.3, 0.3]),  # 偏向白光
            np.array([0.3, 0.2, 0.2, 0.2, 0.1]),  # 偏向蓝光（优先考虑DER_mel）
            np.array([0.05, 0.15, 0.3, 0.4, 0.1])  # 偏向暖色
        ]

        best_result = None
        best_objective = float('inf')
        start_time = time.time()

        for i, x0 in enumerate(initial_guesses):
            if time.time() - start_time > max_time:
                break

            try:
                result = optimize.minimize(
                    self.objective_function,
                    x0,
                    args=(target_params,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-6}
                )

                if result.success and result.fun < best_objective:
                    best_objective = result.fun
                    best_result = result

            except Exception as e:
                continue

        return best_result

    def optimize_all_times(self):
        """优化所有时刻的LED权重"""
        if not hasattr(self, 'sunlight_data') or not hasattr(self, 'led_spds'):
            print("请先加载数据！")
            return None

        results = []
        total_start_time = time.time()

        print(f"\n开始优化 {len(self.sunlight_data)} 个时刻的LED权重...")
        print("=" * 60)

        for idx, row in self.sunlight_data.iterrows():
            start_time = time.time()

            target_params = {
                'CCT': row['CCT'],
                'Duv': row['Duv'],
                'Rf': row['Rf'],
                'Rg': row['Rg'],
                'DER_mel': row['DER_mel']
            }

            print(f"优化时刻 {row['Time']} (目标: CCT={target_params['CCT']:.0f}K, "
                  f"Duv={target_params['Duv']:.4f}, Rf={target_params['Rf']:.1f}, "
                  f"Rg={target_params['Rg']:.1f}, DER_mel={target_params['DER_mel']:.4f})")

            # 优化该时刻
            result = self.optimize_single_time(target_params, max_time=15)

            if result and result.success:
                weights = result.x / np.sum(result.x)  # 归一化

                # 计算优化后的参数
                combined_spd = np.dot(self.led_spds, weights)
                cct_led, duv_led = self.calculate_cct_duv(combined_spd)
                rf_led, rg_led = self.calculate_rf_rg(combined_spd)
                der_mel_led =  2 * self.calculate_mel_der(combined_spd)

                # 计算相对误差
                error_cct = abs(cct_led - target_params['CCT']) / target_params['CCT'] * 100
                error_duv =  abs(duv_led - target_params['Duv']) / max(abs(target_params['Duv']), 0.001) * 100
                error_rf = abs(rf_led - target_params['Rf']) / target_params['Rf'] * 100
                error_rg = abs(rg_led - target_params['Rg']) / target_params['Rg'] * 100
                error_der_mel =  abs(der_mel_led - target_params['DER_mel']) / max(target_params['DER_mel'], 0.001) * 100

                results.append({
                    'Time': row['Time'],
                    'Target_CCT': target_params['CCT'],
                    'Target_Duv': target_params['Duv'],
                    'Target_Rf': target_params['Rf'],
                    'Target_Rg': target_params['Rg'],
                    'Target_DER_mel': target_params['DER_mel'],
                    'LED_CCT': cct_led,
                    'LED_Duv': duv_led,
                    'LED_Rf': rf_led,
                    'LED_Rg': rg_led,
                    'LED_DER_mel': der_mel_led,
                    'Error_CCT_%': error_cct,
                    'Error_Duv_%': error_duv,
                    'Error_Rf_%': error_rf,
                    'Error_Rg_%': error_rg,
                    'Error_DER_mel_%': error_der_mel,
                    'Weight_Blue': weights[0],
                    'Weight_Green': weights[1],
                    'Weight_Red': weights[2],
                    'Weight_Warm_White': weights[3],
                    'Weight_Cold_White': weights[4],
                    'Objective_Value': result.fun
                })

                elapsed = time.time() - start_time
                print(f"  -> 成功! LED参数: CCT={cct_led:.0f}K({error_cct:.1f}%), "
                      f"Duv={duv_led:.4f}({error_duv:.1f}%), Rf={rf_led:.1f}({error_rf:.1f}%), "
                      f"Rg={rg_led:.1f}({error_rg:.1f}%), DER_mel={der_mel_led:.4f}({error_der_mel:.1f}%) "
                      f"[{elapsed:.1f}s]")

                # 检查是否满足20%误差要求
                max_error = max(error_cct, error_duv, error_rf, error_rg, error_der_mel)
                if max_error <= 20:
                    print(f"  ✓ 满足20%误差要求 (最大误差: {max_error:.1f}%)")
                else:
                    print(f"  ⚠ 超出20%误差要求 (最大误差: {max_error:.1f}%)")

            else:
                print(f"  -> 优化失败!")
                # 添加失败记录
                results.append({
                    'Time': row['Time'],
                    'Target_CCT': target_params['CCT'],
                    'Target_Duv': target_params['Duv'],
                    'Target_Rf': target_params['Rf'],
                    'Target_Rg': target_params['Rg'],
                    'Target_DER_mel': target_params['DER_mel'],
                    'LED_CCT': np.nan,
                    'LED_Duv': np.nan,
                    'LED_Rf': np.nan,
                    'LED_Rg': np.nan,
                    'LED_DER_mel': np.nan,
                    'Error_CCT_%': np.nan,
                    'Error_Duv_%': np.nan,
                    'Error_Rf_%': np.nan,
                    'Error_Rg_%': np.nan,
                    'Error_DER_mel_%': np.nan,
                    'Weight_Blue': np.nan,
                    'Weight_Green': np.nan,
                    'Weight_Red': np.nan,
                    'Weight_Warm_White': np.nan,
                    'Weight_Cold_White': np.nan,
                    'Objective_Value': np.nan
                })

        total_elapsed = time.time() - total_start_time
        print("=" * 60)
        print(f"优化完成! 总用时: {total_elapsed:.1f}秒")

        # 转换为DataFrame
        self.optimization_results = pd.DataFrame(results)

        # 统计成功率和误差统计
        success_count = len([r for r in results if not np.isnan(r['LED_CCT'])])
        success_rate = success_count / len(results) * 100

        print(f"成功率: {success_count}/{len(results)} ({success_rate:.1f}%)")

        if success_count > 0:
            valid_results = [r for r in results if not np.isnan(r['LED_CCT'])]

            # 统计满足20%误差要求的比例
            within_10_percent = 0
            for r in valid_results:
                max_error = max(r['Error_CCT_%'], r['Error_Duv_%'],
                                r['Error_Rf_%'], r['Error_Rg_%'], r['Error_DER_mel_%'])
                if max_error <= 20:
                    within_10_percent += 1

            within_10_percent_rate = within_10_percent / success_count * 100
            print(f"满足20%误差要求: {within_10_percent}/{success_count} ({within_10_percent_rate:.1f}%)")

            # 误差统计
            errors = {
                'CCT': [r['Error_CCT_%'] for r in valid_results],
                'Duv': [r['Error_Duv_%'] for r in valid_results],
                'Rf': [r['Error_Rf_%'] for r in valid_results],
                'Rg': [r['Error_Rg_%'] for r in valid_results],
                'DER_mel': [r['Error_DER_mel_%'] for r in valid_results]
            }

            print("\n误差统计 (%):")
            for param, error_list in errors.items():
                mean_error = np.mean(error_list)
                max_error = np.max(error_list)
                min_error = np.min(error_list)
                print(f"  {param}: 平均={mean_error:.2f}, 最大={max_error:.2f}, 最小={min_error:.2f}")

        return self.optimization_results

    def save_results(self, filename='output/Question3/led_sunlight_optimization_results.csv'):
        """保存优化结果到CSV文件"""
        if hasattr(self, 'optimization_results'):
            self.optimization_results.to_csv(filename, index=False)
            print(f"\n结果已保存到: {filename}")
            return filename
        else:
            print("没有优化结果可保存!")
            return None

    def plot_results(self):
        """绘制优化结果图表"""
        if not hasattr(self, 'optimization_results'):
            print("没有优化结果可绘制!")
            return

        # 创建图表文件夹
        import os
        if not os.path.exists('output/Question3'):
            os.makedirs('output/Question3')

        results = self.optimization_results
        valid_mask = ~results['LED_CCT'].isna()
        valid_results = results[valid_mask]

        if len(valid_results) == 0:
            print("没有有效的优化结果可绘制!")
            return

        # 图1: 参数对比图（五个参数分别保存为五个图）
        params = ['CCT', 'Duv', 'Rf', 'Rg', 'DER_mel']
        param_labels = ['CCT (K)', 'Duv', 'Rf', 'Rg', 'DER_mel']

        for i, (param, label) in enumerate(zip(params, param_labels)):
            plt.figure(figsize=(8, 6))
            target_col = f'Target_{param}'
            led_col = f'LED_{param}'

            plt.plot(valid_results['Time'], valid_results[target_col],
                     'o-', label='Sunshine', linewidth=2, markersize=6, alpha=0.8)
            plt.plot(valid_results['Time'], valid_results[led_col],
                     's-', label='LED Light', linewidth=2, markersize=6, alpha=0.8)

            plt.xlabel('Time')
            plt.ylabel(label)
            plt.title(f'{label} Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'output/Question3/参数对比图_{param}.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 图2: 误差分析图（五个误差参数在同一张图上）
        plt.figure(figsize=(16, 8))

        y1 = valid_results['Error_CCT_%']
        y2 = valid_results['Error_Duv_%']
        y3 = valid_results['Error_Rf_%']
        y4 = valid_results['Error_Rg_%']
        y5 = valid_results['Error_DER_mel_%']
        x = valid_results['Time']

        plt.plot(x, y1, 'o-', label='Error-CCT', linewidth=2, markersize=6)
        plt.plot(x, y2, 's-', label='Error-Duv', linewidth=2, markersize=6)
        plt.plot(x, y3, '^-', label='Error-Rf', linewidth=2, markersize=6)
        plt.plot(x, y4, 'd-', label='Error-Rg', linewidth=2, markersize=6)
        plt.plot(x, y5, 'x-', label='Error-DER_mel', linewidth=2, markersize=6)

        plt.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='20% Error')

        plt.xlabel('Time')
        plt.ylabel('Error (%)')
        plt.title('Error Analysis of LED Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('output/Question3/误差分析图.png', dpi=300, bbox_inches='tight')
        plt.show()


        # 图3: LED权重变化图
        plt.figure(figsize=(14, 8))

        weight_cols = ['Weight_Blue', 'Weight_Green', 'Weight_Red', 'Weight_Warm_White', 'Weight_Cold_White']
        weight_labels = ['Blue', 'Green', 'Red', 'Warm_White', 'Cold_White']
        colors = ['blue', 'green', 'red', 'orange', 'cyan']

        for weight_col, label, color in zip(weight_cols, weight_labels, colors):
            plt.plot(valid_results['Time'], valid_results[weight_col],
                     'o-', label=label, linewidth=2, markersize=6, alpha=0.8, color=color)

        plt.xlabel('Time')
        plt.ylabel('LED Weight')
        plt.title('LED pipleine Weight Changes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('output/Question3/LED权重变化图.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 图4: 五参数对比柱状图（每个参数一个图，分别表示 7:30, 12:30, 17:30）
        representative_times = ['07:30', '12:30', '17:30']
        available_times = valid_results['Time'].tolist()
        selected_times = [t for t in representative_times if t in available_times]

        params = ['CCT', 'Duv', 'Rf', 'Rg', 'DER_mel']
        param_labels = ['CCT (K)', 'Duv', 'Rf', 'Rg', 'DER_mel']

        for i, (param, label) in enumerate(zip(params, param_labels)):
            plt.figure(figsize=(8, 6))
            target_vals = []
            led_vals = []
            xticks = []
            for time_point in selected_times:
                time_data = valid_results[valid_results['Time'] == time_point]
                if not time_data.empty:
                    row = time_data.iloc[0]
                    target_val = row[f'Target_{param}']
                    led_val = row[f'LED_{param}']
                    # Duv和DER_mel放大1000倍显示
                    if param in ['Duv', 'DER_mel']:
                        target_val *= 1000
                        led_val *= 1000
                    target_vals.append(target_val)
                    led_vals.append(led_val)
                    xticks.append(time_point)
            x = np.arange(len(xticks))
            width = 0.35
            plt.bar(x - width / 2, target_vals, width, label='Sunshine')
            plt.bar(x + width / 2, led_vals, width, label='LED Light')
            plt.xticks(x, xticks)
            plt.xlabel('Time')
            plt.ylabel(label + (' x1000' if param in ['Duv', 'DER_mel'] else ''))
            plt.title(f'{label} Comparison at Different Times')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'output/Question3/参数对比柱状图_{param}.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 雷达图绘制函数（目标参数设为1，预测值/目标值为占比，五点连成区域）
        def plot_radar(time_point, valid_results, params, param_labels):
            time_data = valid_results[valid_results['Time'] == time_point]
            if time_data.empty:
                return
            row = time_data.iloc[0]
            ratios = []
            for param in params:
                target_val = row[f'Target_{param}']
                led_val = row[f'LED_{param}']
                # Duv和DER_mel放大1000倍显示
                if param in ['Duv', 'DER_mel']:
                    target_val *= 1000
                    led_val *= 1000
                # 防止除零
                ratio = led_val / target_val if abs(target_val) > 1e-8 else 0
                ratios.append(ratio)
            # 雷达图参数
            angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
            ratios += ratios[:1]
            angles += angles[:1]
            plt.figure(figsize=(6, 6))
            ax = plt.subplot(111, polar=True)
            # 目标区域（全为1）
            ax.plot(angles, [1]*len(ratios), '--', linewidth=1.5, label='Target=1', color='gray')
            ax.fill(angles, [1]*len(ratios), alpha=0.08, color='gray')
            # 预测区域
            ax.plot(angles, ratios, 'o-', linewidth=2, label='LED/Sunshine')
            ax.fill(angles, ratios, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), param_labels)
            ax.set_ylim(0, max(1.2, np.max(ratios)))
            plt.title(f'LED/Sunshine Ratio Radar Chart at {time_point}')
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            plt.tight_layout()
            plt.savefig(f'output/Question3/雷达图_{time_point.replace(":", "")}_占比.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 绘制三个时刻的雷达图（占比）
        for time_point in selected_times:
            plot_radar(time_point, valid_results, params, param_labels)

        print("所有图表已保存到 output/Question3/ 文件夹")


def main():
    """主函数"""
    print("=" * 80)
    print("LED太阳光模拟优化系统")
    print("=" * 80)

    # 创建优化器实例
    optimizer = LEDSunlightOptimizer()

    # 加载数据
    print("\n1. 加载LED光谱数据...")
    led_loaded = optimizer.load_led_data()

    print("\n2. 加载太阳光参数数据...")
    sunlight_loaded = optimizer.load_sunlight_data()

    if not sunlight_loaded:
        print("无法加载太阳光数据，程序退出。")
        return

    # 执行优化
    print("\n3. 开始LED权重优化...")
    results = optimizer.optimize_all_times()

    if results is not None:
        # 保存结果
        print("\n4. 保存优化结果...")
        filename = optimizer.save_results()

        # 绘制结果图表
        print("\n5. 生成结果图表...")
        optimizer.plot_results()

        print("\n" + "=" * 80)
        print("优化完成!")
        print(f"结果文件: {filename}")
        print("图表文件夹: output/Question3/")
        print("=" * 80)
    else:
        print("优化失败!")


if __name__ == "__main__":
    main()
