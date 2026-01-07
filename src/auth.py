# -*- coding: utf-8 -*-
"""
用户认证模块
处理登录、注册、Session管理
"""

import streamlit as st
import re
import logging
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_db

logger = logging.getLogger(__name__)


class AuthManager:
    """用户认证管理器"""
    
    def __init__(self):
        self.db = get_db()
    
    @staticmethod
    def init_session():
        """初始化Session状态"""
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'user_info' not in st.session_state:
            st.session_state.user_info = None
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
    
    def login(self, username: str, password: str) -> Tuple[bool, str]:
        """
        用户登录
        
        Returns:
            (成功与否, 消息)
        """
        # 检查登录尝试次数
        if st.session_state.login_attempts >= 5:
            return False, "登录尝试次数过多，请稍后再试"
        
        # 验证输入
        if not username or not password:
            return False, "请输入用户名和密码"
        
        # 验证用户
        user = self.db.verify_user(username, password)
        
        if user:
            st.session_state.logged_in = True
            st.session_state.user_info = user
            st.session_state.login_attempts = 0
            
            # 记录日志
            self.db.add_log(user['id'], '用户登录', f"用户 {username} 登录系统")
            logger.info(f"用户 {username} 登录成功")
            
            return True, f"欢迎回来，{user.get('real_name', username)}！"
        else:
            st.session_state.login_attempts += 1
            logger.warning(f"登录失败: 用户名 {username}")
            return False, "用户名或密码错误"
    
    def logout(self):
        """用户登出"""
        if st.session_state.logged_in and st.session_state.user_info:
            user = st.session_state.user_info
            self.db.add_log(user['id'], '用户登出', f"用户 {user['username']} 退出系统")
            logger.info(f"用户 {user['username']} 登出")
        
        st.session_state.logged_in = False
        st.session_state.user_info = None
    
    def register(self, username: str, password: str, confirm_password: str,
                 real_name: str = None, department: str = None,
                 email: str = None, phone: str = None) -> Tuple[bool, str]:
        """
        用户注册
        
        Returns:
            (成功与否, 消息)
        """
        # 验证用户名
        valid, msg = self.validate_username(username)
        if not valid:
            return False, msg
        
        # 验证密码
        valid, msg = self.validate_password(password)
        if not valid:
            return False, msg
        
        # 确认密码
        if password != confirm_password:
            return False, "两次输入的密码不一致"
        
        # 验证邮箱（如果提供）
        if email:
            valid, msg = self.validate_email(email)
            if not valid:
                return False, msg
        
        # 创建用户
        success, msg = self.db.create_user(
            username=username,
            password=password,
            role='doctor',
            real_name=real_name,
            department=department,
            email=email,
            phone=phone
        )
        
        if success:
            logger.info(f"新用户注册: {username}")
        
        return success, msg
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """验证用户名格式"""
        if not username:
            return False, "用户名不能为空"
        
        if len(username) < 3:
            return False, "用户名长度至少3个字符"
        
        if len(username) > 20:
            return False, "用户名长度不能超过20个字符"
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "用户名只能包含字母、数字和下划线"
        
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """验证密码强度"""
        if not password:
            return False, "密码不能为空"
        
        if len(password) < 6:
            return False, "密码长度至少6个字符"
        
        if len(password) > 50:
            return False, "密码长度不能超过50个字符"
        
        return True, ""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """验证邮箱格式"""
        if not email:
            return True, ""  # 邮箱可选
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "邮箱格式不正确"
        
        return True, ""
    
    @staticmethod
    def is_logged_in() -> bool:
        """检查是否已登录"""
        return st.session_state.get('logged_in', False)
    
    @staticmethod
    def get_current_user() -> Optional[Dict]:
        """获取当前登录用户信息"""
        if AuthManager.is_logged_in():
            return st.session_state.get('user_info')
        return None
    
    @staticmethod
    def is_admin() -> bool:
        """检查当前用户是否是管理员"""
        user = AuthManager.get_current_user()
        return user and user.get('role') == 'admin'
    
    @staticmethod
    def require_login():
        """装饰器：需要登录才能访问的功能"""
        if not AuthManager.is_logged_in():
            st.warning("请先登录后再使用此功能")
            st.stop()
    
    @staticmethod
    def require_admin():
        """装饰器：需要管理员权限"""
        if not AuthManager.is_admin():
            st.error("需要管理员权限才能访问此功能")
            st.stop()


def render_login_page():
    """渲染登录页面"""
    auth = AuthManager()
    auth.init_session()
    
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-header h2 {
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-header">
        <h2>心脏病预测系统</h2>
        <p style="color: #7f8c8d;">基于机器学习的临床决策支持工具</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_login, tab_register = st.tabs(["登录", "注册"])
    
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            
            col1, col2 = st.columns(2)
            with col1:
                remember = st.checkbox("记住我")
            with col2:
                st.markdown("")  # 占位
            
            submit = st.form_submit_button("登录", use_container_width=True)
            
            if submit:
                success, msg = auth.login(username, password)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
            <p>演示账号：admin / admin123（管理员）</p>
            <p>演示账号：doctor / doctor123（医生）</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("用户名", key="reg_username", 
                                         placeholder="3-20位字母、数字或下划线")
            
            col1, col2 = st.columns(2)
            with col1:
                new_password = st.text_input("密码", type="password", key="reg_password",
                                            placeholder="至少6位")
            with col2:
                confirm_password = st.text_input("确认密码", type="password",
                                                placeholder="再次输入密码")
            
            real_name = st.text_input("真实姓名", placeholder="选填")
            department = st.selectbox("科室", 
                ["心内科", "急诊科", "体检中心", "全科", "其他"])
            
            col1, col2 = st.columns(2)
            with col1:
                email = st.text_input("邮箱", placeholder="选填")
            with col2:
                phone = st.text_input("手机号", placeholder="选填")
            
            register = st.form_submit_button("注册", use_container_width=True)
            
            if register:
                success, msg = auth.register(
                    new_username, new_password, confirm_password,
                    real_name, department, email, phone
                )
                if success:
                    st.success("注册成功！请登录")
                else:
                    st.error(msg)


def render_user_info_sidebar():
    """在侧边栏渲染用户信息"""
    auth = AuthManager()
    
    if auth.is_logged_in():
        user = auth.get_current_user()
        role_name = '系统管理员' if user['role'] == 'admin' else '医生'
        
        st.sidebar.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 1.1rem; font-weight: 600;">
                {user.get('real_name', user['username'])}
            </div>
            <div style="font-size: 0.85rem; opacity: 0.9;">
                {role_name} · {user.get('department', '未设置')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("退出登录", use_container_width=True):
            auth.logout()
            st.rerun()
    else:
        st.sidebar.info("未登录")


# 全局认证管理器实例
auth_manager = AuthManager()
