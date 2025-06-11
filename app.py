import streamlit as st
import google.generativeai as genai
# Updated imports to fix deprecation warnings
from langchain_community.llms import GooglePalm
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import json
import re
from datetime import datetime
from typing import Dict, List, Any
import requests
from bs4 import BeautifulSoup

# Konfigurasi Streamlit
st.set_page_config(
    page_title="AI Tutor - Quiz Generator",
    page_icon="🎓",
    layout="wide"
)

class AITutor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.search = DuckDuckGoSearchRun()
        
    def search_topic_info(self, topic: str) -> str:
        """Mencari informasi terbaru tentang topik dari internet"""
        try:
            search_query = f"{topic} tutorial examples latest 2024"
            results = self.search.run(search_query)
            return results[:2000]  # Batasi hasil pencarian
        except Exception as e:
            st.error(f"Error searching: {str(e)}")
            return ""
    
    def generate_quiz(self, topic: str, difficulty: str, num_questions: int) -> Dict:
        """Generate quiz berdasarkan topik dengan berbagai jenis soal"""
        
        # Cari informasi terbaru tentang topik
        search_results = self.search_topic_info(topic)
        
        prompt = f"""
        Sebagai AI Tutor expert, buatkan {num_questions} soal quiz tentang "{topic}" dengan tingkat kesulitan {difficulty}.
        
        Informasi terbaru tentang topik:
        {search_results}
        
        Format setiap soal dalam JSON dengan struktur:
        {{
            "question_id": 1,
            "type": "multiple_choice" | "text_short" | "text_long" | "code",
            "question": "Pertanyaan lengkap",
            "options": ["A", "B", "C", "D"] (hanya untuk multiple choice),
            "correct_answer": "jawaban yang benar",
            "explanation": "penjelasan kenapa jawaban ini benar",
            "points": 10
        }}
        
        Buat variasi jenis soal:
        - 40% multiple choice
        - 30% text short (jawaban singkat)
        - 20% code snippets (untuk topik programming)
        - 10% text long (essay)
        
        Pastikan soal relevan dengan informasi terbaru dan praktis.
        
        Berikan response dalam format JSON array yang valid.
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Ekstrak JSON dari response
            json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
            if json_match:
                quiz_data = json.loads(json_match.group())
                return {"questions": quiz_data, "topic": topic, "difficulty": difficulty}
            else:
                # Fallback jika JSON tidak ditemukan
                return self._create_fallback_quiz(topic, difficulty, num_questions)
        except Exception as e:
            st.error(f"Error generating quiz: {str(e)}")
            return self._create_fallback_quiz(topic, difficulty, num_questions)
    
    def _create_fallback_quiz(self, topic: str, difficulty: str, num_questions: int) -> Dict:
        """Fallback quiz jika generation gagal"""
        fallback_questions = []
        for i in range(num_questions):
            fallback_questions.append({
                "question_id": i + 1,
                "type": "multiple_choice",
                "question": f"Apa konsep dasar dari {topic}? (Soal {i+1})",
                "options": ["Konsep A", "Konsep B", "Konsep C", "Konsep D"],
                "correct_answer": "Konsep A",
                "explanation": f"Konsep A adalah dasar dari {topic}",
                "points": 10
            })
        return {"questions": fallback_questions, "topic": topic, "difficulty": difficulty}
    
    def evaluate_answer(self, question: Dict, user_answer: str) -> Dict:
        """Evaluasi jawaban user dengan AI"""
        
        prompt = f"""
        Evaluasi jawaban user untuk soal berikut:
        
        Soal: {question['question']}
        Jenis: {question['type']}
        Jawaban Benar: {question['correct_answer']}
        Jawaban User: {user_answer}
        
        Berikan evaluasi dalam format JSON:
        {{
            "is_correct": true/false,
            "score": 0-{question['points']},
            "feedback": "feedback detail untuk user",
            "similarity_score": 0-100 (untuk text answers)
        }}
        
        Untuk multiple choice: exact match
        Untuk text answers: evaluasi kesamaan makna dan kelengkapan
        Untuk code: evaluasi logika dan syntax
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Ekstrak JSON dari response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._basic_evaluation(question, user_answer)
        except Exception as e:
            st.error(f"Error evaluating answer: {str(e)}")
            return self._basic_evaluation(question, user_answer)
    
    def _basic_evaluation(self, question: Dict, user_answer: str) -> Dict:
        """Evaluasi dasar jika AI evaluation gagal"""
        if question['type'] == 'multiple_choice':
            is_correct = user_answer.lower() == question['correct_answer'].lower()
            score = question['points'] if is_correct else 0
            return {
                "is_correct": is_correct,
                "score": score,
                "feedback": "Benar!" if is_correct else f"Salah. Jawaban yang benar adalah: {question['correct_answer']}",
                "similarity_score": 100 if is_correct else 0
            }
        else:
            # Untuk text answers, berikan score partial
            similarity = len(set(user_answer.lower().split()) & set(question['correct_answer'].lower().split()))
            score = min(question['points'], similarity * 2)
            return {
                "is_correct": score > question['points'] * 0.6,
                "score": score,
                "feedback": "Jawaban Anda mendekati benar" if score > 0 else "Jawaban masih belum tepat",
                "similarity_score": min(100, similarity * 10)
            }

def init_session_state():
    """Initialize session state variables"""
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'quiz_completed' not in st.session_state:
        st.session_state.quiz_completed = False
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = {}

def main():
    st.title("🎓 AI Tutor - Quiz Generator")
    st.markdown("*Powered by LangChain & Gemini AI*")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar untuk konfigurasi
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # Input API Key
        api_key = st.text_input("Gemini API Key", type="password")
        if not api_key:
            st.warning("Masukkan Gemini API Key untuk melanjutkan")
            st.info("Dapatkan API Key dari: https://makersuite.google.com/app/apikey")
            st.stop()
        
        # Inisialisasi AI Tutor
        if 'tutor' not in st.session_state or st.session_state.get('api_key') != api_key:
            try:
                st.session_state.tutor = AITutor(api_key)
                st.session_state.api_key = api_key
                st.success("✅ AI Tutor berhasil diinisialisasi!")
            except Exception as e:
                st.error(f"❌ Error inisialisasi: {str(e)}")
                st.stop()
        
        # Konfigurasi Quiz
        st.subheader("📝 Konfigurasi Quiz")
        topic = st.text_input("Topik Pembelajaran", value="Python Introduction")
        difficulty = st.selectbox("Tingkat Kesulitan", ["Beginner", "Intermediate", "Advanced"])
        num_questions = st.slider("Jumlah Soal", 3, 10, 5)
        
        generate_button = st.button("🎯 Generate Quiz", type="primary")
    
    # Generate Quiz
    if generate_button and topic:
        with st.spinner("🤖 AI sedang membuat quiz untuk Anda..."):
            try:
                quiz_data = st.session_state.tutor.generate_quiz(topic, difficulty, num_questions)
                st.session_state.quiz_data = quiz_data
                st.session_state.current_question = 0
                st.session_state.user_answers = {}
                st.session_state.quiz_completed = False
                st.session_state.evaluations = {}
                st.success(f"✅ Quiz '{topic}' berhasil dibuat dengan {len(quiz_data['questions'])} soal!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error membuat quiz: {str(e)}")
    
    # Main content area - Welcome screen atau Quiz Interface
    if not st.session_state.quiz_data:
        # Welcome screen
        st.markdown("## 👋 Selamat Datang di AI Tutor!")
        st.markdown("""
        ### 🎯 Fitur Utama:
        - **Quiz Adaptif**: Soal disesuaikan dengan tingkat kesulitan
        - **Berbagai Jenis Soal**: Multiple choice, essay, coding, dan jawaban singkat
        - **Evaluasi AI**: Feedback intelligent menggunakan Gemini AI
        - **Pencarian Real-time**: Informasi terbaru dari internet
        - **Progress Tracking**: Monitor kemajuan belajar Anda
        
        ### 🚀 Cara Menggunakan:
        1. Masukkan **Gemini API Key** di sidebar
        2. Pilih **topik pembelajaran** yang diinginkan
        3. Atur **tingkat kesulitan** dan **jumlah soal**
        4. Klik **Generate Quiz** untuk memulai
        5. Jawab soal dan dapatkan **feedback AI**
        
        ### 📚 Topik Populer:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Programming**\n- Python Basics\n- JavaScript\n- Data Structures\n- Algorithms")
        
        with col2:
            st.info("**Data Science**\n- Machine Learning\n- Statistics\n- Data Analysis\n- Visualization")
        
        with col3:
            st.info("**Web Development**\n- HTML/CSS\n- React\n- Node.js\n- Databases")
    
    # Tampilkan Quiz jika sudah ada
    elif not st.session_state.get('quiz_completed', False):
        quiz = st.session_state.quiz_data
        current_q = st.session_state.current_question
        questions = quiz['questions']
        
        if current_q < len(questions):
            question = questions[current_q]
            
            # Header soal
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader(f"📋 Soal {current_q + 1} dari {len(questions)}")
                st.caption(f"Topik: {quiz['topic']} | Kesulitan: {quiz['difficulty']}")
            with col2:
                st.metric("Poin", question['points'])
            with col3:
                answered = len([k for k in st.session_state.user_answers.keys() if k < len(questions)])
                st.metric("Dijawab", f"{answered}/{len(questions)}")
            
            # Progress bar
            progress = (current_q) / len(questions)
            st.progress(progress, text=f"Progress: {current_q}/{len(questions)} soal")
            
            # Tampilkan soal
            st.markdown("---")
            st.markdown(f"### {question['question']}")
            
            # Badge untuk jenis soal
            type_colors = {
                'multiple_choice': '🔵',
                'text_short': '🟢', 
                'text_long': '🟡',
                'code': '🟣'
            }
            st.markdown(f"{type_colors.get(question['type'], '⚪')} *Jenis: {question['type'].replace('_', ' ').title()}*")
            
            # Input jawaban berdasarkan jenis soal
            user_answer = None
            current_answer = st.session_state.user_answers.get(current_q, "")
            
            if question['type'] == 'multiple_choice':
                # Pre-select if already answered
                default_index = 0
                if current_answer and current_answer in question['options']:
                    default_index = question['options'].index(current_answer)
                user_answer = st.radio("Pilih jawaban:", question['options'], 
                                     index=default_index, key=f"q_{current_q}")
            
            elif question['type'] == 'text_short':
                user_answer = st.text_input("Jawaban singkat:", value=current_answer, 
                                          key=f"q_{current_q}")
            
            elif question['type'] == 'text_long':
                user_answer = st.text_area("Jawaban essay:", value=current_answer, 
                                         key=f"q_{current_q}", height=150)
            
            elif question['type'] == 'code':
                user_answer = st.text_area("Kode program:", value=current_answer, 
                                         key=f"q_{current_q}", height=200)
                if user_answer:
                    st.code(user_answer, language='python')
            
            st.markdown("---")
            
            # Tombol navigasi
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                if current_q > 0:
                    if st.button("⬅️ Sebelumnya", use_container_width=True):
                        if user_answer:
                            st.session_state.user_answers[current_q] = user_answer
                        st.session_state.current_question -= 1
                        st.rerun()
            
            with col2:
                if user_answer and st.button("💾 Simpan Jawaban", use_container_width=True):
                    st.session_state.user_answers[current_q] = user_answer
                    st.success("✅ Jawaban disimpan!")
                    time.sleep(0.5)
            
            with col3:
                if current_q < len(questions) - 1:
                    if st.button("➡️ Selanjutnya", use_container_width=True):
                        if user_answer:
                            st.session_state.user_answers[current_q] = user_answer
                        st.session_state.current_question += 1
                        st.rerun()
                else:
                    if st.button("🏁 Selesai Quiz", use_container_width=True, type="primary"):
                        if user_answer:
                            st.session_state.user_answers[current_q] = user_answer
                        st.session_state.quiz_completed = True
                        st.rerun()
            
            with col4:
                # Jump to question
                jump_to = st.selectbox("Loncat ke soal:", 
                                     range(1, len(questions) + 1), 
                                     index=current_q)
                if jump_to - 1 != current_q:
                    if user_answer:
                        st.session_state.user_answers[current_q] = user_answer
                    st.session_state.current_question = jump_to - 1
                    st.rerun()
    
    # Evaluasi dan Hasil
    if st.session_state.get('quiz_completed', False):
        st.header("📊 Hasil Quiz")
        
        if 'evaluations' not in st.session_state or not st.session_state.evaluations:
            with st.spinner("🤖 AI sedang mengevaluasi jawaban Anda..."):
                evaluations = {}
                questions = st.session_state.quiz_data['questions']
                
                progress_bar = st.progress(0)
                for i, question in enumerate(questions):
                    user_answer = st.session_state.user_answers.get(i, "")
                    evaluation = st.session_state.tutor.evaluate_answer(question, user_answer)
                    evaluations[i] = evaluation
                    progress_bar.progress((i + 1) / len(questions))
                
                st.session_state.evaluations = evaluations
                progress_bar.empty()
        
        # Tampilkan hasil
        total_score = 0
        max_score = 0
        correct_answers = 0
        
        for i, question in enumerate(st.session_state.quiz_data['questions']):
            evaluation = st.session_state.evaluations.get(i, {})
            
            total_score += evaluation.get('score', 0)
            max_score += question['points']
            if evaluation.get('is_correct', False):
                correct_answers += 1
        
        # Summary dengan styling yang lebih baik
        percentage = (total_score/max_score*100) if max_score > 0 else 0
        grade = "A" if percentage >= 90 else "B" if percentage >= 80 else "C" if percentage >= 70 else "D" if percentage >= 60 else "F"
        
        # Grade color
        grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Skor", f"{total_score}/{max_score}")
        with col2:
            st.metric("Persentase", f"{percentage:.1f}%")
        with col3:
            st.metric("Jawaban Benar", f"{correct_answers}/{len(st.session_state.quiz_data['questions'])}")
        with col4:
            st.metric("Grade", f"{grade_colors[grade]} {grade}")
        
        # Performance message
        if percentage >= 90:
            st.success("🎉 Excellent! Pemahaman Anda sangat baik!")
        elif percentage >= 80:
            st.success("👏 Good job! Anda memahami materi dengan baik!")
        elif percentage >= 70:
            st.warning("👍 Not bad! Masih ada ruang untuk improvement.")
        else:
            st.error("💪 Keep learning! Jangan menyerah, terus berlatih!")
        
        # Detail per soal
        st.subheader("📋 Review Jawaban")
        
        for i, question in enumerate(st.session_state.quiz_data['questions']):
            with st.expander(f"Soal {i+1}: {question['question'][:60]}..." if len(question['question']) > 60 else f"Soal {i+1}: {question['question']}"):
                evaluation = st.session_state.evaluations.get(i, {})
                user_answer = st.session_state.user_answers.get(i, "Tidak dijawab")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Jawaban Anda:**")
                    if question['type'] == 'code':
                        st.code(user_answer, language='python')
                    else:
                        st.write(user_answer if user_answer != "Tidak dijawab" else "❌ Tidak dijawab")
                
                with col2:
                    st.write("**Jawaban Benar:**")
                    if question['type'] == 'code':
                        st.code(question['correct_answer'], language='python')
                    else:
                        st.write(question['correct_answer'])
                
                # Feedback dengan styling
                score = evaluation.get('score', 0)
                max_points = question['points']
                
                if evaluation.get('is_correct'):
                    st.success(f"✅ **Benar!** Skor: {score}/{max_points}")
                else:
                    st.error(f"❌ **Salah** Skor: {score}/{max_points}")
                
                # AI Feedback
                feedback = evaluation.get('feedback', 'Tidak ada feedback')
                st.info(f"🤖 **AI Feedback:** {feedback}")
                
                # Explanation
                if 'explanation' in question and question['explanation']:
                    st.markdown(f"📚 **Penjelasan:** {question['explanation']}")
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Quiz Baru", type="primary", use_container_width=True):
                # Reset session state
                for key in ['quiz_data', 'current_question', 'user_answers', 'quiz_completed', 'evaluations']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📥 Download Hasil", use_container_width=True):
                # Create download content
                results_text = f"""
AI TUTOR - HASIL QUIZ
=====================
Topik: {st.session_state.quiz_data['topic']}
Tingkat: {st.session_state.quiz_data['difficulty']}
Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RINGKASAN:
- Total Skor: {total_score}/{max_score}
- Persentase: {percentage:.1f}%
- Grade: {grade}
- Jawaban Benar: {correct_answers}/{len(st.session_state.quiz_data['questions'])}

DETAIL JAWABAN:
"""
                for i, question in enumerate(st.session_state.quiz_data['questions']):
                    evaluation = st.session_state.evaluations.get(i, {})
                    user_answer = st.session_state.user_answers.get(i, "Tidak dijawab")
                    results_text += f"\nSoal {i+1}: {question['question']}\n"
                    results_text += f"Jawaban Anda: {user_answer}\n"
                    results_text += f"Jawaban Benar: {question['correct_answer']}\n"
                    results_text += f"Status: {'Benar' if evaluation.get('is_correct') else 'Salah'}\n"
                    results_text += f"Skor: {evaluation.get('score', 0)}/{question['points']}\n"
                    results_text += f"Feedback: {evaluation.get('feedback', 'Tidak ada feedback')}\n"
                    results_text += "-" * 50 + "\n"
                
                st.download_button(
                    label="📥 Download",
                    data=results_text,
                    file_name=f"quiz_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("🔍 Analisis Detail", use_container_width=True):
                st.info("Fitur analisis detail akan segera hadir!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>🎓 AI Tutor Quiz Generator</strong></p>
        <p>Powered by Google Gemini AI & LangChain | Made with ❤️ using Streamlit</p>
        <p><em>Belajar adaptive dengan teknologi AI terdepan</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
