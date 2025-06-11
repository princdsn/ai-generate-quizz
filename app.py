import streamlit as st
import google.generativeai as genai
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import DuckDuckGoSearchRun
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
        self.model = genai.GenerativeModel('gemini-pro')
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
        fallback_questions = [
            {
                "question_id": 1,
                "type": "multiple_choice",
                "question": f"Apa konsep dasar dari {topic}?",
                "options": ["Konsep A", "Konsep B", "Konsep C", "Konsep D"],
                "correct_answer": "Konsep A",
                "explanation": f"Konsep A adalah dasar dari {topic}",
                "points": 10
            }
        ]
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

def main():
    st.title("🎓 AI Tutor - Quiz Generator")
    st.markdown("*Powered by LangChain & Gemini AI*")
    
    # Sidebar untuk konfigurasi
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        # Input API Key
        api_key = st.text_input("Gemini API Key", type="password")
        if not api_key:
            st.warning("Masukkan Gemini API Key untuk melanjutkan")
            st.stop()
        
        # Inisialisasi AI Tutor
        if 'tutor' not in st.session_state:
            st.session_state.tutor = AITutor(api_key)
        
        # Konfigurasi Quiz
        st.subheader("📝 Konfigurasi Quiz")
        topic = st.text_input("Topik Pembelajaran", value="Python Introduction")
        difficulty = st.selectbox("Tingkat Kesulitan", ["Beginner", "Intermediate", "Advanced"])
        num_questions = st.slider("Jumlah Soal", 3, 10, 5)
        
        generate_button = st.button("🎯 Generate Quiz", type="primary")
    
    # Main content area - Quiz Interface
    
    # Tampilkan Quiz jika sudah ada
    if 'quiz_data' in st.session_state and not st.session_state.get('quiz_completed', False):
        quiz = st.session_state.quiz_data
        current_q = st.session_state.current_question
        questions = quiz['questions']
        
        if current_q < len(questions):
            question = questions[current_q]
            
            # Header soal
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"📋 Soal {current_q + 1} dari {len(questions)}")
            with col2:
                st.metric("Poin", question['points'])
            
            # Progress bar
            progress = (current_q) / len(questions)
            st.progress(progress)
            
            # Tampilkan soal
            st.write(f"**{question['question']}**")
            st.write(f"*Jenis: {question['type'].replace('_', ' ').title()}*")
            
            # Input jawaban berdasarkan jenis soal
            user_answer = None
            
            if question['type'] == 'multiple_choice':
                user_answer = st.radio("Pilih jawaban:", question['options'], key=f"q_{current_q}")
            
            elif question['type'] == 'text_short':
                user_answer = st.text_input("Jawaban singkat:", key=f"q_{current_q}")
            
            elif question['type'] == 'text_long':
                user_answer = st.text_area("Jawaban essay:", key=f"q_{current_q}", height=150)
            
            elif question['type'] == 'code':
                user_answer = st.text_area("Kode program:", key=f"q_{current_q}", height=200)
                st.code(user_answer, language='python')
            
            # Tombol navigasi
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if current_q > 0:
                    if st.button("⬅️ Sebelumnya"):
                        st.session_state.current_question -= 1
                        st.rerun()
            
            with col2:
                if user_answer and st.button("💾 Simpan Jawaban"):
                    st.session_state.user_answers[current_q] = user_answer
                    st.success("Jawaban disimpan!")
            
            with col3:
                if current_q < len(questions) - 1:
                    if st.button("➡️ Selanjutnya"):
                        if user_answer:
                            st.session_state.user_answers[current_q] = user_answer
                        st.session_state.current_question += 1
                        st.rerun()
                else:
                    if st.button("🏁 Selesai"):
                        if user_answer:
                            st.session_state.user_answers[current_q] = user_answer
                        st.session_state.quiz_completed = True
                        st.rerun()
    
    # Evaluasi dan Hasil
    if st.session_state.get('quiz_completed', False):
        st.header("📊 Hasil Quiz")
        
        if 'evaluations' not in st.session_state or not st.session_state.evaluations:
            with st.spinner("🤖 AI sedang mengevaluasi jawaban Anda..."):
                evaluations = {}
                questions = st.session_state.quiz_data['questions']
                
                for i, question in enumerate(questions):
                    user_answer = st.session_state.user_answers.get(i, "")
                    evaluation = st.session_state.tutor.evaluate_answer(question, user_answer)
                    evaluations[i] = evaluation
                
                st.session_state.evaluations = evaluations
        
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
        
        # Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Skor", f"{total_score}/{max_score}")
        with col2:
            st.metric("Persentase", f"{(total_score/max_score*100):.1f}%")
        with col3:
            st.metric("Jawaban Benar", f"{correct_answers}/{len(st.session_state.quiz_data['questions'])}")
        with col4:
            grade = "A" if total_score/max_score >= 0.9 else "B" if total_score/max_score >= 0.8 else "C" if total_score/max_score >= 0.7 else "D"
            st.metric("Grade", grade)
        
        # Detail per soal
        st.subheader("📋 Detail Jawaban")
        
        for i, question in enumerate(st.session_state.quiz_data['questions']):
            with st.expander(f"Soal {i+1}: {question['question'][:50]}..."):
                evaluation = st.session_state.evaluations.get(i, {})
                user_answer = st.session_state.user_answers.get(i, "Tidak dijawab")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Jawaban Anda:**")
                    if question['type'] == 'code':
                        st.code(user_answer, language='python')
                    else:
                        st.write(user_answer)
                
                with col2:
                    st.write("**Jawaban Benar:**")
                    if question['type'] == 'code':
                        st.code(question['correct_answer'], language='python')
                    else:
                        st.write(question['correct_answer'])
                
                # Feedback
                if evaluation.get('is_correct'):
                    st.success(f"✅ Benar! Skor: {evaluation.get('score', 0)}/{question['points']}")
                else:
                    st.error(f"❌ Salah. Skor: {evaluation.get('score', 0)}/{question['points']}")
                
                st.info(f"**Feedback AI:** {evaluation.get('feedback', 'Tidak ada feedback')}")
                
                if 'explanation' in question:
                    st.write(f"**Penjelasan:** {question['explanation']}")
        
        # Tombol reset di hasil
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Buat Quiz Baru", type="primary", use_container_width=True):
                for key in ['quiz_data', 'current_question', 'user_answers', 'quiz_completed', 'evaluations', 'selected_topic']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("*AI Tutor menggunakan teknologi terbaru untuk memberikan pengalaman belajar yang adaptif dan personal.*")

if __name__ == "__main__":
    main()
