# Prompt 1: Direct instruction
PROMPT_1_DIRECT = """
مهمتك: تحسين المنشور التالي مع الالتزام الصارم بالقيود المذكورة.

القيود الإلزامية:
✓ احتفظ باللهجة (فصحى/عامية) كما هي 100%
✓ احتفظ بالتشكيل إن وجد
✓ صحح الأخطاء الإملائية والنحوية فقط
✓ حسّن الوضوح والترابط بين الجمل
✓ لا تغير المعنى أو المحتوى
✓ لا تضف أفكاراً جديدة
✓ حافظ على الطول (± 10% كلمات)
✓ حافظ على النبرة العاطفية الأصلية (حزين/فرح/غاضب/محايد)

القيود المتعلقة بالإخراج:
✓ ابدأ فوراً بعد <START>
✓ أنهِ فوراً قبل <END>
✗ لا تكتب أي شيء قبل <START>
✗ لا تكتب أي شيء بعد <END>
✗ لا تكتب توضيحات أو شروحات
✗ لا تعلق على عملك

المنشور الأصلي:
{post}

المنشور المحسّن (التزم بالقيود أعلاه):
<START>
""".strip()


# Prompt 2: Chain-of-Thought (CoT) prompting
PROMPT_2_COT = """
أنت محرر محتوى متخصص في تحسين منشورات وسائل التواصل الاجتماعي. سأعطيك منشوراً لتحسينه، وعليك اتباع الخطوات التالية:

الخطوة 1: اقرأ المنشور وحدد خصائصه (اللهجة، الأسلوب، الموضوع، المفردات المستخدمة)
الخطوة 2: حدد الأخطاء الإملائية والنحوية إن وجدت
الخطوة 3: حدد الجمل أو العبارات التي تحتاج إلى تحسين في الوضوح أو التماسك
الخطوة 4: قم بإعادة صياغة المنشور مع:
   - إصلاح الأخطاء
   - تحسين الوضوح والتماسك
   - الحفاظ التام على اللهجة والأسلوب الأصلي
   - الحفاظ على عدد الكلمات تقريباً
   - الحفاظ على الطابع العفوي للمنشور

قواعد الإخراج الإلزامية:
- ابدأ المنشور المحسّن مباشرة بعد <START>
- انتهِ بـ <END>
- لا تكتب أي شيء خارج هاتين العلامتين
- لا تشرح التغييرات التي قمت بها

المنشور الأصلي:
{post}

المنشور المحسّن:
<START>
""".strip()


# Prompt 3: English instruction (for cross-lingual robustness testing)
PROMPT_3_ENGLISH = """
You are an expert Arabic language editor specializing in social media content. Your task is to refine and polish the given Arabic post while preserving its original style and character.

Requirements:
- Maintain the original dialect (Modern Standard Arabic or colloquial Arabic)
- Preserve any diacritical marks (tashkeel) if present
- Use vocabulary similar to the original
- Correct spelling and grammatical errors without drastically changing the style
- Improve coherence and clarity while maintaining the spontaneous nature of the post
- Keep the word count approximately the same as the original (±10%)
- Preserve the emotional tone (happy, sad, angry, neutral, etc.)

Critical Output Format Rules:
- Start the refined post immediately after the <START> tag
- End the refined post with the <END> tag
- DO NOT add any comments or notes before <START> or after <END>
- DO NOT write any introductory or concluding phrases
- DO NOT add any text that is not part of the refined post itself
- DO NOT comment on the improvements or changes you made

Original Post:
{post}

Refined Post:
<START>
""".strip()


PROMPTS = {
    "prompt_1_direct": PROMPT_1_DIRECT,
    "prompt_2_cot": PROMPT_2_COT,
    "prompt_3_english": PROMPT_3_ENGLISH,
}
