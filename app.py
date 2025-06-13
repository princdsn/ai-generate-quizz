import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any, Dict
import json
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Streamlit configuration
st.set_page_config(
    page_title="AI Tutor - Quiz Generator",
    page_icon="🎓",
    layout="wide"
)

class GemmaLLM(LLM):
    """Custom LangChain LLM wrapper for Gemma model"""
    
    tokenizer: Any = None
    model: Any = None
    model_name: str = "google/gemma-2b-it"
    max_length: int = 1024
    temperature: float = 0.7
    
    def __init__(self, model_name: str = "google/gemma-2b-it", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        """Load Gemma model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            raise e
    
    @property
    def _llm_type(self) -> str:
        return "gemma"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using Gemma model"""
        try:
            # Format prompt for instruction-tuned model
            formatted_prompt = f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            return "Sorry, I encountered an error generating the response."

class AITutor:
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        self.model_name = model_name
        self.llm = GemmaLLM(model_name=model_name)
        self.search = DuckDuckGoSearchRun()
        
    def search_topic_info(self, topic: str) -> str:
        """Fetch the latest information about a topic from the internet"""
        try:
            query = f"{topic} tutorial examples latest 2024"
            results = self.search.run(query)
            return results[:1500]  # Limit search results
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return ""
    
    def generate_quiz(self, topic: str, difficulty: str, num_questions: int) -> Dict:
        """Generate a quiz based on a topic with varied question types"""
        
        # Fetch latest topic info
        search_results = self.search_topic_info(topic)
        
        prompt = f"""Create {num_questions} quiz questions about "{topic}" at {difficulty} difficulty level.

Context information:
{search_results[:800]}

Generate questions in this exact JSON format:
[
  {{
    "question_id": 1,
    "type": "multiple_choice",
    "question": "What is the main concept of {topic}?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Explanation of why this is correct",
    "points": 10
  }}
]

Question types to include:
- multiple_choice: 40%
- text_short: 30% 
- text_long: 20%
- code: 10% (if programming topic)

Make questions practical and test understanding. Return only valid JSON array."""
        
        try:
            response = self.llm._call(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    quiz_data = json.loads(json_match.group())
                    return {"questions": quiz_data, "topic": topic, "difficulty": difficulty}
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails, create fallback
            return self._create_fallback_quiz(topic, difficulty, num_questions)
            
        except Exception as e:
            st.error(f"Quiz generation error: {str(e)}")
            return self._create_fallback_quiz(topic, difficulty, num_questions)
    
    def _create_fallback_quiz(self, topic: str, difficulty: str, num_questions: int) -> Dict:
        """Fallback quiz if generation fails"""
        fallback_questions = []
        
        question_templates = {
            "multiple_choice": {
                "question": f"What is a key concept in {topic}?",
                "options": ["Fundamental principle", "Basic structure", "Core methodology", "Essential practice"],
                "correct_answer": "Fundamental principle"
            },
            "text_short": {
                "question": f"Briefly explain the importance of {topic}.",
                "correct_answer": f"{topic} is important because it provides essential knowledge and skills."
            },
            "text_long": {
                "question": f"Describe in detail how {topic} can be applied in real-world scenarios.",
                "correct_answer": f"{topic} has many practical applications including problem-solving and implementation."
            },
            "code": {
                "question": f"Write a simple code example demonstrating {topic}.",
                "correct_answer": f"# Example code for {topic}\nprint('Hello {topic}')"
            }
        }
        
        types = ["multiple_choice", "text_short", "text_long", "code"]
        
        for i in range(num_questions):
            q_type = types[i % len(types)]
            template = question_templates[q_type]
            
            question = {
                "question_id": i + 1,
                "type": q_type,
                "question": template["question"],
                "correct_answer": template["correct_answer"],
                "explanation": f"This covers fundamental aspects of {topic}.",
                "points": 10
            }
            
            if q_type == "multiple_choice":
                question["options"] = template["options"]
            
            fallback_questions.append(question)
        
        return {"questions": fallback_questions, "topic": topic, "difficulty": difficulty}
    
    def evaluate_answer(self, question: Dict, user_answer: str) -> Dict:
        """Evaluate a user's answer using Gemma"""
        
        prompt = f"""Evaluate this quiz answer:

Question: {question['question']}
Question Type: {question['type']}
Correct Answer: {question['correct_answer']}
User's Answer: {user_answer}

Provide evaluation in this format:
- Is the answer correct? (Yes/No)
- Score out of {question['points']}
- Brief feedback explaining why
- Similarity percentage for text answers

Be fair and consider partial credit for text answers."""
        
        try:
            response = self.llm._call(prompt)
            
            # Parse response to extract evaluation
            is_correct = False
            score = 0
            feedback = "Unable to evaluate"
            similarity_score = 0
            
            response_lower = response.lower()
            
            # Simple evaluation logic
            if question['type'] == 'multiple_choice':
                is_correct = user_answer.lower().strip() == question['correct_answer'].lower().strip()
                score = question['points'] if is_correct else 0
                similarity_score = 100 if is_correct else 0
            else:
                # For text answers, check for keyword matches
                user_words = set(user_answer.lower().split())
                correct_words = set(question['correct_answer'].lower().split())
                common_words = user_words.intersection(correct_words)
                
                if len(correct_words) > 0:
                    similarity_score = len(common_words) / len(correct_words) * 100
                    score = int(question['points'] * similarity_score / 100)
                    is_correct = similarity_score > 60
            
            if "correct" in response_lower or "yes" in response_lower:
                is_correct = True
            elif "incorrect" in response_lower or "no" in response_lower:
                is_correct = False
            
            feedback = response[:200] if response else "Basic evaluation completed"
            
            return {
                "is_correct": is_correct,
                "score": max(0, min(question['points'], score)),
                "feedback": feedback,
                "similarity_score": max(0, min(100, similarity_score))
            }
            
        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")
            return self._basic_evaluation(question, user_answer)
    
    def _basic_evaluation(self, question: Dict, user_answer: str) -> Dict:
        """Basic evaluation if AI fails"""
        if question['type'] == 'multiple_choice':
            is_correct = user_answer.lower() == question['correct_answer'].lower()
            score = question['points'] if is_correct else 0
            return {
                "is_correct": is_correct,
                "score": score,
                "feedback": "Correct!" if is_correct else f"Incorrect. The correct answer is: {question['correct_answer']}",
                "similarity_score": 100 if is_correct else 0
            }
        else:
            words_match = len(set(user_answer.lower().split()) & set(question['correct_answer'].lower().split()))
            total_words = len(set(question['correct_answer'].lower().split()))
            similarity = (words_match / total_words * 100) if total_words > 0 else 0
            score = int(question['points'] * similarity / 100)
            
            return {
                "is_correct": similarity > 60,
                "score": score,
                "feedback": f"Similarity: {similarity:.1f}%. " + ("Good match!" if similarity > 60 else "Could be improved."),
                "similarity_score": similarity
            }


def init_session_state():
    """Initialize Streamlit session state variables"""
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
    st.markdown("*Powered by LangChain & Gemma AI*")
    
    init_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_options = [
            "google/gemma-2b-it",
            "google/gemma-7b-it",
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it"
        ]
        
        selected_model = st.selectbox(
            "Select Gemma Model",
            model_options,
            index=0,
            help="Choose the Gemma model variant. Larger models may be more accurate but slower."
        )
        
        if 'tutor' not in st.session_state or st.session_state.get('model_name') != selected_model:
            with st.spinner(f"Loading {selected_model}..."):
                try:
                    st.session_state.tutor = AITutor(model_name=selected_model)
                    st.session_state.model_name = selected_model
                    st.success("✅ AI Tutor initialized successfully!")
                except Exception as e:
                    st.error(f"Initialization error: {str(e)}")
                    st.error("Make sure you have enough memory and the model is available.")
                    st.stop()
        
        st.subheader("📝 Quiz Setup")
        topic = st.text_input("Learning Topic", value="Python Introduction")
        difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])
        num_questions = st.slider("Number of Questions", 3, 10, 5)
        generate_button = st.button("🎯 Generate Quiz")
        
        # System info
        st.subheader("💻 System Info")
        st.info(f"CUDA Available: {'Yes' if torch.cuda.is_available() else 'No'}")
        if torch.cuda.is_available():
            st.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # Generate quiz
    if generate_button and topic:
        with st.spinner("🤖 Generating your quiz..."):
            quiz_data = st.session_state.tutor.generate_quiz(topic, difficulty, num_questions)
            st.session_state.quiz_data = quiz_data
            st.session_state.current_question = 0
            st.session_state.user_answers = {}
            st.session_state.quiz_completed = False
            st.session_state.evaluations = {}
            st.success(f"✅ Quiz '{topic}' created with {len(quiz_data['questions'])} questions!")
            st.rerun()

    # Display welcome or quiz UI
    if not st.session_state.quiz_data:
        st.markdown("## 👋 Welcome to AI Tutor!")
        st.markdown("""
### 🎯 Key Features:
- **Gemma AI**: Powered by Google's Gemma language model
- **Adaptive Quizzes**: Questions tailored by difficulty
- **Multiple Question Types**: MCQ, essay, coding, short answer
- **AI Evaluation**: Intelligent feedback by Gemma AI
- **Real-time Search**: Latest topic info from the web
- **Progress Tracking**: Monitor your learning journey

### 🚀 How to Use:
1. Select your preferred **Gemma model** in the sidebar
2. Choose a **learning topic**
3. Set **difficulty** and **number of questions**
4. Click **Generate Quiz**
5. Answer questions and get **AI feedback**

### 📚 Popular Topics:""")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Programming**\n- Python Basics\n- JavaScript\n- Data Structures\n- Algorithms")
        with col2:
            st.info("**Data Science**\n- Machine Learning\n- Statistics\n- Data Analysis\n- Visualization")
        with col3:
            st.info("**Web Development**\n- HTML/CSS\n- React\n- Node.js\n- Databases")

    # Quiz interface
    elif not st.session_state.quiz_completed:
        quiz = st.session_state.quiz_data
        idx = st.session_state.current_question
        q_list = quiz['questions']
        question = q_list[idx]

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader(f"📋 Question {idx+1} of {len(q_list)}")
            st.caption(f"Topic: {quiz['topic']} | Difficulty: {quiz['difficulty']}")
        with col2:
            st.metric("Points", question['points'])
        with col3:
            answered = len([i for i in st.session_state.user_answers if i < len(q_list)])
            st.metric("Answered", f"{answered}/{len(q_list)}")

        progress = idx / len(q_list)
        st.progress(progress, text=f"Progress: {idx}/{len(q_list)} questions")
        st.markdown("---")
        st.markdown(f"### {question['question']}")
        type_icons = {'multiple_choice': '🔵', 'text_short': '🟢', 'text_long': '🟡', 'code': '🟣'}
        st.markdown(f"{type_icons.get(question['type'], '')} *Type: {question['type'].replace('_',' ').title()}*")

        user_answer = None
        prev = st.session_state.user_answers.get(idx, "")
        if question['type'] == 'multiple_choice':
            options = question.get('options', ['A', 'B', 'C', 'D'])
            default = options.index(prev) if prev in options else 0
            user_answer = st.radio("Select an option:", options, index=default, key=f"q_{idx}")
        elif question['type'] == 'text_short':
            user_answer = st.text_input("Short answer:", value=prev, key=f"q_{idx}")
        elif question['type'] == 'text_long':
            user_answer = st.text_area("Essay response:", value=prev, key=f"q_{idx}", height=150)
        else:
            user_answer = st.text_area("Code answer:", value=prev, key=f"q_{idx}", height=200)
            if user_answer:
                st.code(user_answer, language='python')
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if idx > 0 and st.button("⬅️ Previous", use_container_width=True):
                st.session_state.user_answers[idx] = user_answer
                st.session_state.current_question -= 1
                st.rerun()
        with col2:
            if user_answer and st.button("💾 Save Answer", use_container_width=True):
                st.session_state.user_answers[idx] = user_answer
                st.success("Answer saved!")
        with col3:
            if idx < len(q_list)-1 and st.button("➡️ Next", use_container_width=True):
                st.session_state.user_answers[idx] = user_answer
                st.session_state.current_question += 1
                st.rerun()
            elif idx == len(q_list)-1 and st.button("🏁 Finish Quiz", use_container_width=True, type="primary"):
                st.session_state.user_answers[idx] = user_answer
                st.session_state.quiz_completed = True
                st.rerun()
        with col4:
            jump = st.selectbox("Jump to question:", range(1, len(q_list)+1), index=idx)
            if jump-1 != idx:
                st.session_state.user_answers[idx] = user_answer
                st.session_state.current_question = jump-1
                st.rerun()

    # Evaluation and results
    if st.session_state.quiz_completed:
        st.header("📊 Quiz Results")
        if not st.session_state.evaluations:
            with st.spinner("🤖 Evaluating answers..."):
                evals = {}
                total = len(st.session_state.quiz_data['questions'])
                progress_bar = st.progress(0)
                for i, q in enumerate(st.session_state.quiz_data['questions']):
                    ua = st.session_state.user_answers.get(i, "")
                    evals[i] = st.session_state.tutor.evaluate_answer(q, ua)
                    progress_bar.progress((i+1)/total)
                st.session_state.evaluations = evals
                progress_bar.empty()

        total_score = sum(e['score'] for e in st.session_state.evaluations.values())
        max_score = sum(q['points'] for q in st.session_state.quiz_data['questions'])
        correct_count = sum(1 for e in st.session_state.evaluations.values() if e['is_correct'])
        pct = (total_score / max_score * 100) if max_score>0 else 0
        grade = 'A' if pct>=90 else 'B' if pct>=80 else 'C' if pct>=70 else 'D' if pct>=60 else 'F'
        icons = {'A':'🟢','B':'🔵','C':'🟡','D':'🟠','F':'🔴'}
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Score", f"{total_score}/{max_score}")
        with col2: st.metric("Percentage", f"{pct:.1f}%")
        with col3: st.metric("Correct Answers", f"{correct_count}/{len(st.session_state.quiz_data['questions'])}")
        with col4: st.metric("Grade", f"{icons[grade]} {grade}")

        if pct>=90: st.success("🎉 Excellent work! You have a strong grasp of the material.")
        elif pct>=80: st.success("👏 Good job! You understand the content well.")
        elif pct>=70: st.warning("👍 Not bad! There's room for improvement.")
        else: st.error("💪 Keep practicing! Don't give up.")

        st.subheader("📋 Review Responses")
        for i, q in enumerate(st.session_state.quiz_data['questions']):
            with st.expander(f"Q{i+1}: {q['question'][:60]}{'...' if len(q['question'])>60 else ''}"):
                ua = st.session_state.user_answers.get(i, "Not answered")
                ev = st.session_state.evaluations.get(i, {})
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Your Answer:**")
                    if q['type']=='code': st.code(ua, language='python')
                    else: st.write(ua)
                with c2:
                    st.write("**Correct Answer:**")
                    if q['type']=='code': st.code(q['correct_answer'], language='python')
                    else: st.write(q['correct_answer'])
                if ev.get('is_correct'): st.success(f"✅ Correct! Score: {ev['score']}/{q['points']}")
                else: st.error(f"❌ Incorrect. Score: {ev['score']}/{q['points']}")
                st.info(f"🤖 AI Feedback: {ev.get('feedback','No feedback')}")
                if 'explanation' in q: st.markdown(f"📚 Explanation: {q['explanation']}")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 New Quiz", type="primary", use_container_width=True):
                for key in ['quiz_data','current_question','user_answers','quiz_completed','evaluations']:
                    st.session_state.pop(key, None)
                st.rerun()
        with c2:
            if st.button("📥 Download Results", use_container_width=True):
                output = [f"AI TUTOR - QUIZ RESULTS", "="*20,
                         f"Topic: {st.session_state.quiz_data['topic']}",
                         f"Difficulty: {st.session_state.quiz_data['difficulty']}",
                         f"Model: {st.session_state.get('model_name', 'gemma-2b-it')}",
                         f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                         "", f"Summary:",
                         f"- Total Score: {total_score}/{max_score}",
                         f"- Percentage: {pct:.1f}%", f"- Grade: {grade}",
                         f"- Correct Answers: {correct_count}/{len(st.session_state.quiz_data['questions'])}",
                         "", "Details:"]
                for i, q in enumerate(st.session_state.quiz_data['questions']):
                    ev = st.session_state.evaluations[i]
                    ua = st.session_state.user_answers.get(i, "Not answered")
                    output += ["", f"Question {i+1}: {q['question']}",
                              f"Your answer: {ua}",
                              f"Correct answer: {q['correct_answer']}",
                              f"Status: {'Correct' if ev['is_correct'] else 'Incorrect'}",
                              f"Score: {ev['score']}/{q['points']}",
                              f"Feedback: {ev['feedback']}", "-"*30]
                text_data = "\n".join(output)
                st.download_button("📥 Download", data=text_data,
                                   file_name=f"quiz_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                   mime="text/plain")
        with c3:
            if st.button("🔍 Detailed Analysis", use_container_width=True):
                st.info("Detailed analysis feature coming soon!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
<table style='width:100%; text-align:center; color:#666;'>
  <tr><td><strong>🎓 AI Tutor Quiz Generator</strong></td></tr>
  <tr><td>Powered by Gemma AI & LangChain | Made with ❤️ using Streamlit</td></tr>
  <tr><td><em>Adaptive learning with cutting-edge AI technology</em></td></tr>
</table>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
