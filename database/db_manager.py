# -*- coding: utf-8 -*-
"""
数据库管理模块
封装所有 CRUD 操作（增删改查）
"""

import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DatabaseConfig, PathConfig


class DBManager:
    """数据库管理器类 - 单例模式"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.db_path = DatabaseConfig.DB_PATH
        self._init_database()
        self._initialized = True
    
    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'doctor',
                real_name TEXT,
                department TEXT,
                email TEXT,
                phone TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # 患者预测历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER,
                patient_name TEXT,
                patient_id_card TEXT,
                gender INTEGER,
                age INTEGER,
                heart_rate REAL,
                systolic_bp REAL,
                diastolic_bp REAL,
                blood_sugar REAL,
                ck_mb REAL,
                troponin REAL,
                predict_result INTEGER,
                predict_proba REAL,
                risk_level TEXT,
                model_used TEXT,
                shap_values TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doctor_id) REFERENCES users(id)
            )
        ''')
        
        # 系统日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # 模型训练记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                auc_roc REAL,
                parameters TEXT,
                training_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_best INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        
        # 创建默认账户
        self._create_default_users(cursor, conn)
        
        conn.close()
    
    def _create_default_users(self, cursor, conn):
        """创建默认用户账户"""
        # 创建管理员
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", 
                      (DatabaseConfig.DEFAULT_ADMIN['username'],))
        if cursor.fetchone()[0] == 0:
            admin = DatabaseConfig.DEFAULT_ADMIN
            password_hash = self._hash_password(admin['password'])
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, real_name, department)
                VALUES (?, ?, ?, ?, ?)
            ''', (admin['username'], password_hash, admin['role'], 
                  admin['real_name'], admin['department']))
        
        # 创建演示医生账户
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?",
                      (DatabaseConfig.DEFAULT_DOCTOR['username'],))
        if cursor.fetchone()[0] == 0:
            doctor = DatabaseConfig.DEFAULT_DOCTOR
            password_hash = self._hash_password(doctor['password'])
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, real_name, department)
                VALUES (?, ?, ?, ?, ?)
            ''', (doctor['username'], password_hash, doctor['role'],
                  doctor['real_name'], doctor['department']))
        
        conn.commit()
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """密码哈希（SHA-256）"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    # ==================== 用户管理 ====================
    
    def create_user(self, username: str, password: str, role: str = 'doctor',
                    real_name: str = None, department: str = None,
                    email: str = None, phone: str = None) -> Tuple[bool, str]:
        """
        创建新用户
        
        Returns:
            (成功与否, 消息)
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            password_hash = self._hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, real_name, department, email, phone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, password_hash, role, real_name, department, email, phone))
            conn.commit()
            conn.close()
            return True, f"用户 {username} 创建成功"
        except sqlite3.IntegrityError:
            return False, "用户名已存在"
        except Exception as e:
            return False, f"创建失败: {str(e)}"
    
    def verify_user(self, username: str, password: str) -> Optional[Dict]:
        """
        验证用户登录
        
        Returns:
            用户信息字典或None
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        password_hash = self._hash_password(password)
        
        cursor.execute('''
            SELECT id, username, role, real_name, department, email, phone
            FROM users
            WHERE username = ? AND password_hash = ? AND is_active = 1
        ''', (username, password_hash))
        
        row = cursor.fetchone()
        
        if row:
            # 更新最后登录时间
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?',
                          (datetime.now(), row['id']))
            conn.commit()
            user_info = dict(row)
            conn.close()
            return user_info
        
        conn.close()
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """根据ID获取用户信息"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, role, real_name, department, email, phone, 
                   created_at, last_login, is_active
            FROM users WHERE id = ?
        ''', (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_all_users(self) -> List[Dict]:
        """获取所有用户列表"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, role, real_name, department, email, phone,
                   created_at, last_login, is_active
            FROM users ORDER BY created_at DESC
        ''')
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return users
    
    def update_user_status(self, user_id: int, is_active: bool) -> bool:
        """更新用户状态（启用/禁用）"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_active = ? WHERE id = ?',
                      (int(is_active), user_id))
        conn.commit()
        conn.close()
        return True
    
    def change_password(self, user_id: int, new_password: str) -> bool:
        """修改用户密码"""
        conn = self.get_connection()
        cursor = conn.cursor()
        password_hash = self._hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?',
                      (password_hash, user_id))
        conn.commit()
        conn.close()
        return True
    
    # ==================== 患者预测历史 ====================
    
    def save_prediction(self, doctor_id: int, patient_data: Dict,
                       predict_result: int, predict_proba: float,
                       risk_level: str, model_used: str = "RandomForest",
                       shap_values: str = None, notes: str = None) -> int:
        """
        保存预测记录
        
        Returns:
            记录ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patients_history 
            (doctor_id, patient_name, patient_id_card, gender, age, heart_rate,
             systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin,
             predict_result, predict_proba, risk_level, model_used, shap_values, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doctor_id,
            patient_data.get('patient_name', '未命名'),
            patient_data.get('patient_id_card', ''),
            patient_data.get('gender', 0),
            patient_data.get('age', 0),
            patient_data.get('heart_rate', 0),
            patient_data.get('systolic_bp', 0),
            patient_data.get('diastolic_bp', 0),
            patient_data.get('blood_sugar', 0),
            patient_data.get('ck_mb', 0),
            patient_data.get('troponin', 0),
            predict_result,
            predict_proba,
            risk_level,
            model_used,
            shap_values,
            notes
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id
    
    def get_prediction_history(self, doctor_id: int = None, 
                              limit: int = 100,
                              risk_filter: str = None) -> pd.DataFrame:
        """
        获取预测历史记录
        
        Args:
            doctor_id: 医生ID（None表示获取所有）
            limit: 返回记录数限制
            risk_filter: 风险筛选（'高风险'/'低风险'/None）
        """
        conn = self.get_connection()
        
        query = '''
            SELECT ph.*, u.username, u.real_name as doctor_name
            FROM patients_history ph
            LEFT JOIN users u ON ph.doctor_id = u.id
            WHERE 1=1
        '''
        params = []
        
        if doctor_id:
            query += ' AND ph.doctor_id = ?'
            params.append(doctor_id)
        
        if risk_filter == '高风险':
            query += ' AND ph.predict_result = 1'
        elif risk_filter == '低风险':
            query += ' AND ph.predict_result = 0'
        
        query += ' ORDER BY ph.created_at DESC LIMIT ?'
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_prediction_by_id(self, record_id: int) -> Optional[Dict]:
        """获取单条预测记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients_history WHERE id = ?', (record_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_prediction_statistics(self, doctor_id: int = None) -> Dict:
        """
        获取预测统计数据
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        where_clause = "WHERE doctor_id = ?" if doctor_id else ""
        params = (doctor_id,) if doctor_id else ()
        
        # 总预测数
        cursor.execute(f'SELECT COUNT(*) FROM patients_history {where_clause}', params)
        total = cursor.fetchone()[0]
        
        # 高风险数
        if doctor_id:
            cursor.execute(f'''
                SELECT COUNT(*) FROM patients_history 
                WHERE doctor_id = ? AND predict_result = 1
            ''', (doctor_id,))
        else:
            cursor.execute('SELECT COUNT(*) FROM patients_history WHERE predict_result = 1')
        high_risk = cursor.fetchone()[0]
        
        # 今日预测数
        if doctor_id:
            cursor.execute(f'''
                SELECT COUNT(*) FROM patients_history 
                WHERE doctor_id = ? AND DATE(created_at) = DATE('now')
            ''', (doctor_id,))
        else:
            cursor.execute('''
                SELECT COUNT(*) FROM patients_history 
                WHERE DATE(created_at) = DATE('now')
            ''')
        today = cursor.fetchone()[0]
        
        # 平均概率
        cursor.execute(f'SELECT AVG(predict_proba) FROM patients_history {where_clause}', params)
        avg_prob = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_predictions': total,
            'high_risk_count': high_risk,
            'low_risk_count': total - high_risk,
            'today_predictions': today,
            'high_risk_ratio': high_risk / total if total > 0 else 0,
            'avg_probability': avg_prob
        }
    
    def search_similar_cases(self, patient_data: Dict, k: int = 3) -> pd.DataFrame:
        """
        基于KNN检索相似病例
        
        Args:
            patient_data: 当前患者数据
            k: 返回的相似病例数量
        """
        conn = self.get_connection()
        
        # 获取所有历史病例
        df = pd.read_sql_query('''
            SELECT * FROM patients_history 
            WHERE age IS NOT NULL AND ck_mb IS NOT NULL AND troponin IS NOT NULL
        ''', conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        # 计算欧氏距离（使用关键特征）
        features = ['age', 'ck_mb', 'troponin', 'gender']
        
        # 归一化处理
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        if len(df) < 2:
            return df.head(k)
        
        df_features = df[features].fillna(0)
        df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=features)
        
        # 当前患者特征
        current = pd.DataFrame([{
            'age': patient_data.get('age', 0),
            'ck_mb': patient_data.get('ck_mb', 0),
            'troponin': patient_data.get('troponin', 0),
            'gender': patient_data.get('gender', 0)
        }])
        current_scaled = scaler.transform(current)
        
        # 计算距离
        import numpy as np
        distances = np.sqrt(((df_scaled.values - current_scaled) ** 2).sum(axis=1))
        df['distance'] = distances
        
        # 返回最相似的k个
        similar = df.nsmallest(k, 'distance')
        return similar
    
    # ==================== 系统日志 ====================
    
    def add_log(self, user_id: int, action: str, details: str = None,
                ip_address: str = None):
        """添加系统日志"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO logs (user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (user_id, action, details, ip_address))
        conn.commit()
        conn.close()
    
    def get_logs(self, limit: int = 100, user_id: int = None) -> List[Dict]:
        """获取系统日志"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT l.*, u.username, u.real_name
                FROM logs l
                LEFT JOIN users u ON l.user_id = u.id
                WHERE l.user_id = ?
                ORDER BY l.created_at DESC LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT l.*, u.username, u.real_name
                FROM logs l
                LEFT JOIN users u ON l.user_id = u.id
                ORDER BY l.created_at DESC LIMIT ?
            ''', (limit,))
        
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return logs
    
    # ==================== 统计方法 ====================
    
    def get_system_stats(self) -> Dict:
        """
        获取系统统计数据
        
        Returns:
            包含总预测数、总用户数、阳性率等的字典
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 总预测数
        cursor.execute('SELECT COUNT(*) FROM patients_history')
        total_predictions = cursor.fetchone()[0]
        
        # 总用户数
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        total_users = cursor.fetchone()[0]
        
        # 阳性数
        cursor.execute('SELECT COUNT(*) FROM patients_history WHERE predict_result = 1')
        positive_count = cursor.fetchone()[0]
        
        # 计算阳性率
        positive_rate = (positive_count / total_predictions * 100) if total_predictions > 0 else 0
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'total_users': total_users,
            'positive_count': positive_count,
            'positive_rate': positive_rate
        }
    
    def get_today_stats(self) -> Dict:
        """
        获取今日统计数据
        
        Returns:
            包含今日预测数、今日高危人数等的字典
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 今日预测数
        cursor.execute('''
            SELECT COUNT(*) FROM patients_history 
            WHERE DATE(created_at) = DATE('now', 'localtime')
        ''')
        today_predictions = cursor.fetchone()[0]
        
        # 今日高危人数
        cursor.execute('''
            SELECT COUNT(*) FROM patients_history 
            WHERE DATE(created_at) = DATE('now', 'localtime') AND predict_result = 1
        ''')
        high_risk_count = cursor.fetchone()[0]
        
        # 今日低风险人数
        cursor.execute('''
            SELECT COUNT(*) FROM patients_history 
            WHERE DATE(created_at) = DATE('now', 'localtime') AND predict_result = 0
        ''')
        low_risk_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'today_predictions': today_predictions,
            'high_risk_count': high_risk_count,
            'low_risk_count': low_risk_count
        }
    
    def get_week_stats(self) -> Dict:
        """
        获取本周统计数据
        
        Returns:
            包含本周预测数、高危人数、环比等的字典
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 本周预测数
        cursor.execute('''
            SELECT COUNT(*) FROM patients_history 
            WHERE DATE(created_at) >= DATE('now', 'localtime', '-7 days')
        ''')
        week_predictions = cursor.fetchone()[0]
        
        # 本周高危人数
        cursor.execute('''
            SELECT COUNT(*) FROM patients_history 
            WHERE DATE(created_at) >= DATE('now', 'localtime', '-7 days') AND predict_result = 1
        ''')
        week_high_risk = cursor.fetchone()[0]
        
        # 上周预测数（用于计算环比）
        cursor.execute('''
            SELECT COUNT(*) FROM patients_history 
            WHERE DATE(created_at) >= DATE('now', 'localtime', '-14 days')
              AND DATE(created_at) < DATE('now', 'localtime', '-7 days')
        ''')
        last_week_predictions = cursor.fetchone()[0]
        
        # 计算环比增长率
        if last_week_predictions > 0:
            growth_rate = (week_predictions - last_week_predictions) / last_week_predictions * 100
        else:
            growth_rate = 100 if week_predictions > 0 else 0
        
        conn.close()
        
        return {
            'week_predictions': week_predictions,
            'week_high_risk': week_high_risk,
            'last_week_predictions': last_week_predictions,
            'growth_rate': growth_rate
        }
    
    # ==================== 模型记录 ====================
    
    def save_model_record(self, model_name: str, metrics: Dict,
                         parameters: str = None, training_time: float = None,
                         is_best: bool = False):
        """保存模型训练记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if is_best:
            cursor.execute('UPDATE model_records SET is_best = 0')
        
        cursor.execute('''
            INSERT INTO model_records 
            (model_name, accuracy, precision_score, recall, f1_score, auc_roc,
             parameters, training_time, is_best)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1', 0),
            metrics.get('auc_roc', 0),
            parameters,
            training_time,
            int(is_best)
        ))
        conn.commit()
        conn.close()
    
    def get_model_records(self, limit: int = 50) -> pd.DataFrame:
        """获取模型训练记录"""
        conn = self.get_connection()
        df = pd.read_sql_query('''
            SELECT * FROM model_records ORDER BY created_at DESC LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        return df


# 全局单例
db_manager = DBManager()


def get_db() -> DBManager:
    """获取数据库管理器实例"""
    return db_manager
