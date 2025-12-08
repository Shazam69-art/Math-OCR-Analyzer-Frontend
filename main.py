import os
import base64
import io
import json
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import google.generativeai as genai
from typing import List
import logging
from datetime import datetime
import re
from fastapi import Request
import asyncio

# Configure Gemini from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CAS Education Math OCR Analyzer API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

def pil_to_base64_png(im: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

async def process_uploaded_file(file: UploadFile) -> List[str]:
    """Process uploaded file and return base64 encoded pages."""
    content = await file.read()
 
    if file.content_type == "application/pdf":
        logger.warning("PDF processing requires pypdfium2. Install it for full functionality.")
        raise HTTPException(status_code=400, detail="PDF processing is not available. Please install pypdfium2 or convert PDFs to images.")
 
    elif file.content_type.startswith("image/"):
        try:
            image = Image.open(io.BytesIO(content))
            return [pil_to_base64_png(image)]
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

async def transcribe_image(file: UploadFile) -> str:
    """Transcribe a single image using Gemini asynchronously."""
    try:
        content = await file.read()
        base64_str = base64.b64encode(content).decode('utf-8')
        image_part = {
            "mime_type": file.content_type or "image/png",
            "data": base64_str
        }
        prompt = "Transcribe this image exactly as written, converting all mathematical expressions to LaTeX format. Preserve the exact structure, text, and steps verbatim. For handwritten content, transcribe word-for-word and symbol-for-symbol without any interpretation or correction. Ignore strikethrough text completely."
        response = await model.generate_content_async([prompt, image_part])
        return response.text.strip()
    except Exception as e:
        logger.error(f"Transcription error for file {file.filename}: {str(e)}")
        return f"Transcription failed for {file.filename}: {str(e)}"

async def transcribe_images(files: List[UploadFile]) -> List[str]:
    """Transcribe multiple images in parallel using asyncio.gather."""
    tasks = [transcribe_image(file) for file in files]
    return await asyncio.gather(*tasks)

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/style.css")
async def serve_css():
    return FileResponse("style.css")

@app.post("/analyze-chat")
async def analyze_chat(
        message: str = Form(""),
        question_count: int = Form(0),
        files: List[UploadFile] = File([])
):
    """Main analysis endpoint using Gemini with IMPROVED OUTPUT FORMAT."""
    try:
        logger.info(f"Analysis request - Message: {message[:100]}, Files: {len(files)}, Question count: {question_count}")
      
        # Split files into questions and solutions based on question_count
        question_files = files[:question_count]
        solution_files = files[question_count:]
      
        # Transcribe images asynchronously to get text
        question_texts = await transcribe_images(question_files)
        solution_texts = await transcribe_images(solution_files)
      
        # Format transcribed texts
        question_paper = "\n\n".join([f"Question Page {i+1}:\n{text}" for i, text in enumerate(question_texts)])
        solution_paper = "\n\n".join([f"Solution Page {i+1}:\n{text}" for i, text in enumerate(solution_texts)])
      
        # UPDATED & FINAL SYSTEM PROMPT (WITH RULES 9–11 FOR STRICTER ACCURACY)
        system_prompt = r"""You are a **PhD-Level Math Teacher** analyzing student work based on transcribed texts.
**CRITICAL INSTRUCTIONS FOR OUTPUT (FOLLOW STRICTLY TO AVOID TIMEOUTS - BE CONCISE, NO EXTRA TEXT):**
1. **ALL MATHEMATICAL EXPRESSIONS MUST BE IN LATEX/MATHJAX FORMAT** - Use $...$ for inline math and $$...$$ for display math. Ensure 100% proper LaTeX for rendering. Keep output short to process quickly.
2. **STUDENT'S SOLUTION: 100% VERBATIM TRANSCRIPTION ONLY** - Copy EXACTLY from the transcribed solution text. DO NOT add, modify, regenerate, interpret, or invent ANY content. If unclear, copy as-is. Ignore strikethrough completely. NO additions like 'The student wrote...' - just the raw steps.
3. **ERROR ANALYSIS: EXTREMELY SHORT, MATH-FOCUSED (1-5 WORDS + MATHJAX)** - Use minimal English, focus on math terms. Example: "Step 2: Wrong \(\frac{du}{dx} = 2x\) (should be \(2\))". NO long sentences, explanations, or corrections here. Max 10 words per error.
4. **CORRECT SOLUTION: 100% ACCURATE, STEP-BY-STEP** - Provide precise, error-free steps leading to the correct final answer. Ensure mathematical rigor.
5. **SEPARATE EACH QUESTION CLEARLY** - Analyze one question at a time based on labels in transcriptions.
6. **MARK AS CORRECT** if final answer matches, even if steps differ slightly.
7. **ONLY FLAG ERRORS** for significant mathematical issues affecting the answer.
8. **BE EFFICIENT** - Short responses to avoid timeouts. Focus only on key elements.
9. **IF NO SOLUTION PROVIDED** - Clearly state "No solution provided" in the student's solution section and mark as incorrect if no answer is given.
10. **IF PARTIAL ANSWER** - State "Partial answer given" and analyze what's there.
11. **IF ANSWER DOESN'T MATCH** - Mark as incorrect regardless of steps.
**OUTPUT FORMAT - FOLLOW EXACTLY (NO DEVIATIONS):**
## Question [EXACT LABEL]:
**Full Question:** [Exact transcribed question in MathJax]
### Student's Solution – Exact Copy:
**Step 1:** [Exact transcribed line 1 in MathJax - VERBATIM]
**Step 2:** [Exact transcribed line 2 in MathJax - VERBATIM]
...
### Error Analysis:
**Step X:** [Short math term error, e.g., "Invalid \(u\)-sub: \(\sqrt{x} \neq x^{1/2}\)" ]
...
### Corrected Solution:
**Step 1:** [Correct math step in MathJax]
...
**Final Answer:** $$\boxed{final_answer}$$
---
**PERFORMANCE TABLE (UPDATE BASED ON ACTUAL ERRORS FOUND - KEEP SHORT)**
| Concept No. | Concept (With Explanation) | Example | Status |
|-------------|----------------------------|---------|--------|
| 1 | Basic Formulas | Standard Formula of Integration | **Performance:** Not Tested |
| 2 | Application of Formulae | \(\int x^9 dx = \frac{x^{10}}{10} + C\) | **Performance:** Not Tested |
| 3 | Basic Trigonometric Ratios Integration | Integration of \(\sin x, \cos x, \tan x, \sec x, \cot x, \csc x\) | **Performance:** Not Tested |
| 4 | Basic Squares & Cubes Trigonometric Ratios Integration | \(\int \tan^2 x dx, \int \cot^2 x dx, \int \sin^2 2x dx, \int \cos^2 2x dx\) | **Performance:** Not Tested |
| 5 | Integration of Linear Functions via Substitution | \(\int (3x+5)^7 dx, \int (4-9x)^5 dx, \int \sec^2 (3x+5) dx\) | **Performance:** Not Tested |
| 6 | Basic Substitution (level 1) | \(\int \frac{\log x}{x} dx, \int \frac{\sec^2 (\log x)}{x} dx, \int \frac{e^{\tan^{-1}x}}{1+x^2} dx, \int \frac{\sin \sqrt{x}}{\sqrt{x}} dx\) | **Performance:** Not Tested |
| 7 | Substitution (Some Simplification Involved) (level 2) | \(\int \frac{2x}{(2x+1)^2} dx, \int \frac{2+3x}{3-2x} dx\) | **Performance:** Not Tested |
| 8 | Complex Substitution (Some Simplification Involved) (level 3) | \(\int \frac{dx}{x \sqrt{x^6 - 1}}, \int \frac{x^2 \tan^{-1} x^3}{1+x^6} dx\) | **Performance:** Not Tested |
| 9 | Substitution with Square root | \(\int \frac{x-1}{\sqrt{x+4}} dx, \int x \sqrt{x+2} dx\) | **Performance:** Not Tested |
| 10 | Same order Integration (Solving by adding and subtraction) | \(\int \frac{3x^2}{1+x^2} dx\) | **Performance:** Not Tested |
| 11 | Using formulae & completing the square methods<br/>(i) \(\int \frac{dx}{a^2 - x^2} = \frac{1}{2a} \log \left\lvert \frac{a + x}{a - x} \right\rvert + C\)<br/>(ii) \(\int \frac{dx}{x^2 - a^2} = \frac{1}{2a} \log \left\lvert \frac{x - a}{x + a} \right\rvert + C\) | \(\int \frac{dx}{x^2 + 8x + 20}\) | **Performance:** Not Tested |
| 12 | Standard Integrals<br/>(i) \(\int \frac{dx}{\sqrt{a^2 - x^2}} = \sin^{-1} \frac{x}{a} + C\)<br/>(ii) \(\int \frac{dx}{\sqrt{x^2 - a^2}} = \log \left\lvert x + \sqrt{x^2 - a^2} \right\rvert + C\)<br/>(iii) \(\int \frac{dx}{\sqrt{x^2 + a^2}} = \log \left\lvert x + \sqrt{x^2 + a^2} \right\rvert + C\) | **Evaluate:**<br/>(i) \(\int \frac{dx}{\sqrt{9 - 25x^2}}\)<br/>(ii) \(\int \frac{dx}{\sqrt{x^2 - 3x + 2}}\) | **Performance:** Not Tested |
| 13 | Integration of Linear In Numerator and Quadratic (or Sq Root of Quadratic) In Denominator.<br/>Integrals of the form:<br/>\( \int \frac{(px + q)}{\sqrt{(ax^2 + bx + c)}} dx \)<br/>\( \int \frac{(px + q)}{(ax^2 + bx + c)} dx \) | \(\int \frac{(5x + 3)}{\sqrt{x^2 + 4x + 10}} dx\)<br/>\( \int \frac{(2x + 1)}{(4 - 3x - x^2)} dx\) | **Performance:** Not Tested |
| 14 | By Parts (ILATE)<br/>\( \int (uv) dx = u \int v dx - \int \left( \frac{du}{dx} \int v dx \right) dx \) | (i) \(\int x \sec^2 x dx\) | **Performance:** Not Tested |
| 15 | By Part - In which "1" has to be taken as one of the functions to start solving. | \(\int \log x dx\)<br/>(ii) \(\int \tan^{-1} x dx\) | **Performance:** Not Tested |
| 16 | Inverse Trigonometric By Parts | (ii) \(\int \tan^{-1} x dx\) | **Performance:** Not Tested |
| 17 | Integrals of the form \(\int e^x [f(x) + f'(x)] dx\) | (ii) \(\int e^x \left( \frac{1}{x^2} - \frac{2}{x^3} \right) dx\) | **Performance:** Not Tested |
| 18 | Integration of (e^x)(sinx)<br/>Where terms keeps on repeating.<br/>\( \int e^{2x} \sin x dx \) | \(\int e^{3x} \sin 4x dx\)<br/>\( \int e^{3x} \sin 4x dx\) | **Performance:** Not Tested |
**UPDATE TABLE BASED ON ACTUAL ANALYSIS:** For each concept tested, update status like: "Performance: Tested 2 Times - Perfect 2 (Q.1, Q.3)" or "Performance: Tested 1 Time - Mistakes 1 (Q.2)". Keep concise.
## Performance Insights
[Short insights with MathJax where needed - max 3-5 sentences]"""

        # Prepare content for Gemini
        contents = [
            system_prompt,
            f"Question Paper Transcription:\n{question_paper}",
            f"Student Solution Transcription:\n{solution_paper}"
        ]
        if message:
            contents.append(f"User request: {message}")
      
        # Call Gemini asynchronously
        try:
            response = await model.generate_content_async(contents)
            ai_response = response.text
        except Exception as genai_error:
            logger.error(f"Gemini API error: {str(genai_error)}")
            raise HTTPException(status_code=504, detail="Analysis service is taking longer than expected. Please try again.")
      
        # Parse detailed data for frontend
        detailed_data = parse_detailed_data_improved(ai_response)
        logger.info(f"Analysis completed. Found {len(detailed_data.get('questions', []))} questions")
      
        return JSONResponse({
            "status": "success",
            "response": ai_response,
            "detailed_data": detailed_data,
            "files_processed": [f"Processed {len(question_files)} questions and {len(solution_files)} solutions"]
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def parse_detailed_data_improved(response_text):
    """UPDATED parsing: Handles shorter errors, stricter verbatim steps, improved regex for quality."""
    questions = []
    if not response_text:
        return {"questions": questions}
 
    question_sections = re.split(r'## Question\s+', response_text)
    question_sections = [section for section in question_sections if section.strip() and not section.startswith('Questions found')]
 
    for i, section in enumerate(question_sections, 1):
        try:
            question_id_match = re.search(r'^([A-Z]?[0-9]+[a-z]?(?:\([a-z]\))?[^:\n]*):?', section)
            if question_id_match:
                question_id = question_id_match.group(1).strip()
            else:
                alt_match = re.search(r'^(Q?[0-9]+[a-z]?(?:\s*\([a-z]\))?)', section)
                question_id = alt_match.group(1).strip() if alt_match else f"Q{i}"
         
            question_text = "Question content not extracted"
            if '**Full Question:**' in section:
                question_part = section.split('**Full Question:**')[1]
                if '###' in question_part:
                    question_text = question_part.split('###')[0].strip()
                else:
                    question_text = question_part.strip()
            question_text = re.sub(r'### Student\'s Solution.*?###', '', question_text, flags=re.DOTALL).strip()
         
            steps = []
            if '### Student\'s Solution' in section:
                solution_part = section.split('### Student\'s Solution')[1]
                if '###' in solution_part:
                    solution_section = solution_part.split('###')[0]
                else:
                    solution_section = solution_part
                step_pattern = r'\*\*Step\s+(\d+):\*\*\s*(.*?)(?=\*\*Step\s+\d+:|###|\Z)'
                step_matches = re.findall(step_pattern, solution_section, re.DOTALL | re.IGNORECASE)
                steps = [match[1].strip() for match in step_matches if match[1].strip()]
            if not steps:
                steps = ["No solution provided"]
         
            mistakes = []
            has_errors = False
            if '### Error Analysis' in section:
                error_part = section.split('### Error Analysis')[1]
                if '###' in error_part:
                    error_section = error_part.split('###')[0]
                else:
                    error_section = error_part
                error_pattern = r'\*\*Step\s*(\d+):\*\*\s*(.*?)(?=\*\*Step\s*\d+:|\Z)'
                error_matches = re.findall(error_pattern, error_section, re.DOTALL | re.IGNORECASE)
                for match in error_matches:
                    step_num, error_desc = match
                    if error_desc.strip():
                        has_errors = True
                        mistakes.append({
                            "step": step_num,
                            "status": "Error",
                            "desc": error_desc.strip()
                        })
         
            corrected_steps = []
            if '### Corrected Solution' in section:
                correct_part = section.split('### Corrected Solution')[1]
                if '##' in correct_part:
                    correct_section = correct_part.split('##')[0]
                else:
                    correct_section = correct_part
                step_pattern = r'\*\*Step\s+(\d+):\*\*\s*(.*?)(?=\*\*Step\s+\d+:|\*\*Final Answer|\Z)'
                step_matches = re.findall(step_pattern, correct_section, re.DOTALL)
                corrected_steps = [match[1].strip() for match in step_matches if match[1].strip()]
         
            final_answer = ""
            final_match = re.search(r'\\boxed{(.*?)}', section)
            if final_match:
                final_answer = f"$$\\boxed{{{final_match.group(1)}}}$$"
            elif '**Final Answer:**' in section:
                answer_part = section.split('**Final Answer:**')[1]
                if '\\boxed' in answer_part:
                    boxed_match = re.search(r'\\boxed{(.*?)}', answer_part)
                    final_answer = f"$$\\boxed{{{boxed_match.group(1)}}}$$" if boxed_match else ""
                else:
                    final_answer_text = answer_part.split('\n')[0].strip()
                    final_answer = f"$${final_answer_text}$$" if final_answer_text else ""
         
            questions.append({
                "id": question_id,
                "questionText": question_text[:500] + "..." if len(question_text) > 500 else question_text,
                "steps": steps,
                "mistakes": mistakes,
                "hasErrors": has_errors,
                "correctedSteps": corrected_steps or ["Complete solution will be shown after analysis"],
                "finalAnswer": final_answer or "Answer will be determined after analysis"
            })
     
        except Exception as e:
            logger.error(f"Error parsing question {i}: {e}")
            questions.append({
                "id": f"Q{i}",
                "questionText": f"Question {i}",
                "steps": ["Analysis in progress"],
                "mistakes": [],
                "hasErrors": False,
                "correctedSteps": ["Solution analysis"],
                "finalAnswer": "Answer pending"
            })
 
    return {"questions": questions}

@app.post("/analyze-feedback")
async def analyze_feedback(request: dict):
    try:
        question = request.get("question", {})
        feedback = request.get("feedback", "")
        original_analysis = request.get("original_analysis", "")
     
        if not question or not feedback:
            return JSONResponse({
                "success": False,
                "error": "Missing question or feedback data"
            })
     
        feedback_prompt = f"""
        A user has provided feedback on the analysis of Question {question.get('id', 'Unknown')}:
        Original Question: {question.get('questionText', '')}
        User Feedback: {feedback}
        Please re-analyze this specific question considering the user's feedback.
        Focus on:
        1. Addressing the user's specific concerns
        2. Providing clearer mathematical explanations
        3. Ensuring all mathematical expressions are in proper MathJax/LaTeX format
        Provide the updated analysis for this question only.
        """
     
        response = model.generate_content(feedback_prompt)
        updated_analysis = response.text
     
        updated_question = parse_single_question(updated_analysis, question.get('id', f"Q{len(question)}"))
     
        return JSONResponse({
            "success": True,
            "updated_question": updated_question,
            "message": "Analysis updated successfully"
        })
 
    except Exception as e:
        logger.error(f"Feedback analysis failed: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": f"Feedback analysis failed: {str(e)}"
        }, status_code=500)

def parse_single_question(analysis_text, question_id):
    return {
        "id": question_id,
        "questionText": extract_question_text(analysis_text),
        "steps": extract_steps(analysis_text),
        "mistakes": extract_mistakes(analysis_text),
        "hasErrors": True,
        "correctedSteps": extract_corrected_steps(analysis_text),
        "finalAnswer": extract_final_answer(analysis_text)
    }

def extract_question_text(text):
    match = re.search(r'Original Question:\s*(.*?)(?=User Feedback|$)', text, re.DOTALL)
    return match.group(1).strip() if match else "Question text not available"

def extract_steps(text):
    return ["Step analysis in progress"]

def extract_mistakes(text):
    return [{"step": 1, "status": "Error", "desc": "Re-analyzed based on user feedback"}]

def extract_corrected_steps(text):
    return ["Corrected solution based on user feedback"]

def extract_final_answer(text):
    match = re.search(r'\\boxed{(.*?)}', text)
    return f"$$\\boxed{{{match.group(1)}}}$$" if match else "Final answer pending"

@app.post("/create-practice-paper")
async def create_practice_paper(request: dict):
    try:
        detailed_data = request.get("detailed_data", {})
        questions_with_errors = []
     
        for q in detailed_data.get("questions", []):
            if q.get('hasErrors', False) and q.get('mistakes'):
                questions_with_errors.append(q)
     
        logger.info(f"Found {len(questions_with_errors)} questions with errors for practice paper")
     
        if not questions_with_errors:
            return JSONResponse({
                "success": False,
                "error": "No questions with errors found. Your solutions appear to be correct!"
            })
     
        practice_prompt = f"""Create a targeted practice paper with EXACTLY {len(questions_with_errors)} redesigned questions.
**CRITICAL REQUIREMENTS:**
1. For EACH original question, create ONE modified practice question - redesign ALL provided.
2. **PRESERVE THE EXACT QUESTION NUMBER/LABEL from the original** (e.g., if original is "1(a)", use "1(a)" in the output)
3. Focus on the SAME concepts where errors occurred
4. ALL math expressions MUST be in LaTeX/MathJax format ($...$ or $$...$$) for proper rendering.
5. Output ONLY the questions in the specified format below - NO additional text, tables, or commentary. Ensure every question is redesigned.
**QUESTIONS TO REDESIGN (ALL MUST BE INCLUDED):**
{format_questions_for_practice_prompt(questions_with_errors)}
**OUTPUT FORMAT - USE THIS EXACTLY FOR EACH QUESTION:**
### Based on Question [EXACT_QUESTION_NUMBER]
**Original Question:**
[Copy the exact original question in MathJax]
**Modified Question:**
[Modified version testing SAME concepts with different values in MathJax]
---
**EXAMPLE FORMAT:**
### Based on Question 1(a)
**Original Question:**
Evaluate $\int x^9 dx$
**Modified Question:**
Evaluate $\int x^7 dx$
---
**IMPORTANT:**
- Use the EXACT question number from the original (1(a), 2(b), 3, etc.)
- NO extra text - just "Based on Question"
- Each question must be separated by ---
- Focus on same mathematical concepts with different coefficients/values - redesign EVERY one provided"""
     
        response = model.generate_content(practice_prompt)
        practice_paper = response.text
     
        logger.info(f"Successfully generated practice paper with {len(questions_with_errors)} questions")
     
        return JSONResponse({
            "success": True,
            "practice_paper": practice_paper,
            "questions_used": len(questions_with_errors),
            "message": f"Practice paper created targeting {len(questions_with_errors)} error areas"
        })
 
    except Exception as e:
        logger.error(f"Practice paper creation failed: {str(e)}")
        return JSONResponse({
            "success": False,
            "error": f"Practice paper creation failed: {str(e)}"
        }, status_code=500)

def format_questions_for_practice_prompt(questions_with_errors):
    formatted = ""
    for q in questions_with_errors:
        formatted += f"\n**Question {q['id']}:** {q['questionText'][:200]}...\n"
        formatted += f"**Errors Found:** {', '.join([m.get('desc', '')[:100] for m in q.get('mistakes', [])])}\n"
    return formatted

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


