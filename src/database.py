# -*- coding: utf-8 -*-
"""
数据库操作模块
使用SQLite进行数据持久化，存储用户信息、预测历史和系统日志
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd


class DatabaseManager:
    """数据库管理器类"""
    
    def __init__(self, db_path: str = "heart_disease.db"):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        ''')
        
        # 预测历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                patient_name TEXT,
                gender INTEGER,
                age INTEGER,
                heart_rate REAL,
                systolic_bp REAL,
                diastolic_bp REAL,
                blood_sugar REAL,
                ck_mb REAL,
                troponin REAL,
                prediction_result INTEGER,
                probability REAL,
                risk_level TEXT,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # 系统日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
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
        
        # 创建默认管理员账户
        self._create_default_admin(cursor, conn)
        
        conn.close()
    
    def _create_default_admin(self, cursor, conn):
        """创建默认管理员账户"""
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            password_hash = self._hash_password("admin123")
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, real_name, department)
                VALUES (?, ?, ?, ?, ?)
            ''', ('admin', password_hash, 'admin', '系统管理员', '信息科'))
            conn.commit()
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    # ==================== 用户管理 ====================
    
    def create_user(self, username: str, password: str, role: str = 'doctor',
                    real_name: str = None, department: str = None) -> bool:
        """
        创建新用户
        
        Args:
            username: 用户名
            password: 密码
            role: 角色 (admin/doctor)
            real_name: 真实姓名
            department: 科室
            
        Returns:
            是否创建成功
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            password_hash = self._hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, real_name, department)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, password_hash, role, real_name, department))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def verify_user(self, username: str, password: str) -> Optional[Dict]:
        """
        验证用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            用户信息字典或None
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        password_hash = self._hash_password(password)
        cursor.execute('''
            SELECT id, username, role, real_name, department
            FROM users
            WHERE username = ? AND password_hash = ? AND is_active = 1
        ''', (username, password_hash))
        row = cursor.fetchone()
        
        if row:
            # 更新最后登录时间
            cursor.execute('''
                UPDATE users SET last_login = ? WHERE id = ?
            ''', (datetime.now(), row['id']))
            conn.commit()
            
            user_info = dict(row)
            conn.close()
            return user_info
        
        conn.close()
        return None
    
    def get_all_users(self) -> List[Dict]:
        """获取所有用户列表"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, role, real_name, department, created_at, last_login, is_active
            FROM users ORDER BY created_at DESC
        ''')
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return users
    
    def update_user_status(self, user_id: int, is_active: bool) -> bool:
        """更新用户状态"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE users SET is_active = ? WHERE id = ?', (int(is_active), user_id))
        conn.commit()
        conn.close()
        return True
    
    # ==================== 预测历史管理 ====================
    
    def save_prediction(self, user_id: int, patient_data: Dict, 
                       prediction_result: int, probability: float,
                       risk_level: str, model_used: str = "RandomForest") -> int:
        """
        保存预测记录
        
        Args:
            user_id: 用户ID
            patient_data: 患者数据字典
            prediction_result: 预测结果 (0/1)
            probability: 预测概率
            risk_level: 风险等级
            model_used: 使用的模型名称
            
        Returns:
            记录ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_history 
            (user_id, patient_name, gender, age, heart_rate, systolic_bp, 
             diastolic_bp, blood_sugar, ck_mb, troponin, prediction_result, 
             probability, risk_level, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            patient_data.get('patient_name', '未命名'),
            patient_data.get('gender', 0),
            patient_data.get('age', 0),
            patient_data.get('heart_rate', 0),
            patient_data.get('systolic_bp', 0),
            patient_data.get('diastolic_bp', 0),
            patient_data.get('blood_sugar', 0),
            patient_data.get('ck_mb', 0),
            patient_data.get('troponin', 0),
            prediction_result,
            probability,
            risk_level,
            model_used
        ))
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id
    
    def get_prediction_history(self, user_id: int = None, limit: int = 100) -> pd.DataFrame:
        """
        获取预测历史记录
        
        Args:
            user_id: 用户ID（None表示获取所有）
            limit: 返回记录数限制
            
        Returns:
            预测历史DataFrame
        """
        conn = self.get_connection()
        
        if user_id:
            query = '''
                SELECT ph.*, u.username, u.real_name as doctor_name
                FROM prediction_history ph
                LEFT JOIN users u ON ph.user_id = u.id
                WHERE ph.user_id = ?
                ORDER BY ph.created_at DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(user_id, limit))
        else:
            query = '''
                SELECT ph.*, u.username, u.real_name as doctor_name
                FROM prediction_history ph
                LEFT JOIN users u ON ph.user_id = u.id
                ORDER BY ph.created_at DESC
                LIMIT ?
            '''
            df = pd.read_sql_query(query, conn, params=(limit,))
        
        conn.close()
        return df
    
    def get_prediction_statistics(self, user_id: int = None) -> Dict:
        """
        获取预测统计数据
        
        Args:
            user_id: 用户ID（None表示全部统计）
            
        Returns:
            统计数据字典
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        where_clause = "WHERE user_id = ?" if user_id else ""
        params = (user_id,) if user_id else ()
        
        # 总预测数
        cursor.execute(f'SELECT COUNT(*) FROM prediction_history {where_clause}', params)
        total_predictions = cursor.fetchone()[0]
        
        # 高风险预测数
        cursor.execute(f'''
            SELECT COUNT(*) FROM prediction_history 
            {where_clause} {"AND" if user_id else "WHERE"} prediction_result = 1
        ''', params)
        high_risk_count = cursor.fetchone()[0]
        
        # 今日预测数
        cursor.execute(f'''
            SELECT COUNT(*) FROM prediction_history 
            {where_clause} {"AND" if user_id else "WHERE"} DATE(created_at) = DATE('now')
        ''', params)
        today_predictions = cursor.fetchone()[0]
        
        # 平均概率
        cursor.execute(f'SELECT AVG(probability) FROM prediction_history {where_clause}', params)
        avg_probability = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'high_risk_count': high_risk_count,
            'low_risk_count': total_predictions - high_risk_count,
            'today_predictions': today_predictions,
            'high_risk_ratio': high_risk_count / total_predictions if total_predictions > 0 else 0,
            'avg_probability': avg_probability
        }
    
    # ==================== 系统日志管理 ====================
    
    def add_log(self, user_id: int, action: str, details: str = None, ip_address: str = None):
        """添加系统日志"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_logs (user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (user_id, action, details, ip_address))
        conn.commit()
        conn.close()
    
    def get_logs(self, limit: int = 100) -> List[Dict]:
        """获取系统日志"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sl.*, u.username
            FROM system_logs sl
            LEFT JOIN users u ON sl.user_id = u.id
            ORDER BY sl.created_at DESC
            LIMIT ?
        ''', (limit,))
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return logs
    
    # ==================== 模型记录管理 ====================
    
    def save_model_record(self, model_name: str, metrics: Dict, 
                         parameters: str = None, training_time: float = None,
                         is_best: bool = False):
        """保存模型训练记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 如果是最佳模型，先清除其他最佳标记
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
            SELECT * FROM model_records
            ORDER BY created_at DESC
            LIMIT ?
        ''', conn, params=(limit,))
        conn.close()
        return df


# 单例模式
_db_instance = None

def get_database() -> DatabaseManager:
    """获取数据库管理器单例"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
