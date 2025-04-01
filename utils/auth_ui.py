import streamlit as st
from utils.auth_service import UserService

def init_auth():
    """Initialize session state variables for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"  # Can be 'login' or 'register'

def auth_page():
    """Display and handle authentication page (login and register)."""
    init_auth()
    user_service = UserService()
    
    st.title("🇮🇳 Indian Financial Assistant")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Login", key="login_button", type="primary" if st.session_state.auth_page == "login" else "secondary"):
            st.session_state.auth_page = "login"
            st.rerun()
    
    with col2:
        if st.button("Register", key="register_button", type="primary" if st.session_state.auth_page == "register" else "secondary"):
            st.session_state.auth_page = "register"
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.auth_page == "login":
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    result = user_service.login_user(username, password)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.session_state.authenticated = True
                        st.session_state.current_user = result["user"]
                        st.success("Login successful!")
                        st.rerun()
    else:
        st.subheader("Register")
        with st.form("register_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            st.markdown("""
            **Password requirements:**
            - Minimum 8 characters
            - Maximum 26 characters
            - At least one uppercase letter
            - At least one lowercase letter
            - At least one number
            - At least one special character (!@#$%^&*())
            """)
            
            submit_button = st.form_submit_button("Register")
            
            if submit_button:
                if not username or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    result = user_service.create_user(username, password)
                    if isinstance(result, dict) and "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("Registration successful! You can now login.")
                        st.session_state.auth_page = "login"
                        st.rerun()

def logout():
    """Logout current user."""
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.rerun()