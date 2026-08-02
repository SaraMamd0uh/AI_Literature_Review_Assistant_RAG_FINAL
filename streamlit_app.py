
import { motion } from "framer-motion";
import { ArrowLeft, BookOpen, ShieldCheck } from "lucide-react";
import heroImg from "@/assets/hero-social-work.jpg";
import { Award, Lightbulb, Search, Target } from "lucide-react";
import heroImg from "@/assets/hero-books.jpg";

const Hero = () => {

  return (
    <section className="relative pt-32 pb-20 overflow-hidden paper-dots">
      <div className="absolute -top-32 -left-24 w-96 h-96 rounded-full bg-accent/20 blur-3xl pointer-events-none" />
      <div className="max-w-6xl mx-auto px-5 grid lg:grid-cols-2 gap-12 items-center relative">
    <section className="relative pt-28 sm:pt-32 pb-16 overflow-hidden">
      <div className="max-w-6xl mx-auto px-5 grid lg:grid-cols-2 gap-10 lg:gap-6 items-center">
        <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <span className="eyebrow">
            <span className="w-6 h-px bg-accent" />
            منصّة معرفية للمؤسسات
          </span>
          <h1 className="mt-4 text-4xl sm:text-5xl lg:text-[3.4rem] leading-[1.15] text-foreground">
            مساعد ذكي متخصص في
            <span className="block text-primary">الخدمة الاجتماعية وعلم الاجتماع</span>
          <h1 className="text-4xl sm:text-5xl lg:text-[3.6rem] font-extrabold leading-[1.2] text-foreground">
            اعثر على الإجابة
            <span className="block">التي تبحث عنها في</span>
            <span className="block text-primary">الخدمة الاجتماعية.</span>
          </h1>
          <p className="mt-5 text-base sm:text-lg text-muted-foreground leading-relaxed max-w-xl">
            اسأل عن أي مفهوم، نظرية، أداة تدخل مهني أو منهج بحث — واحصل على إجابة دقيقة مستندة إلى
            مكتبة من الكتب والمراجع الأكاديمية، مع ذكر المصدر ورقم الصفحة في كل إجابة.
          <p className="mt-5 text-sm sm:text-base text-muted-foreground leading-relaxed max-w-md">
            مساعد ذكي يجيبك من كتب ومراجع الخدمة الاجتماعية وعلم الاجتماع، مع ذكر المصدر ورقم الصفحة.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
          <div className="mt-8 flex items-center gap-2 bg-card rounded-full p-2 pr-5 border border-border shadow-soft max-w-md">
            <Search className="w-4 h-4 text-muted-foreground shrink-0" />
            <input
              readOnly
              onFocus={() => go("chat")}
              placeholder="اكتب سؤالك المهني هنا…"
              className="flex-1 min-w-0 bg-transparent text-sm outline-none text-foreground placeholder:text-muted-foreground"
            />
            <button
              onClick={() => go("chat")}
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-primary text-primary-foreground font-bold text-sm hover:opacity-90 transition-opacity shadow-soft"
              className="shrink-0 px-6 py-3 rounded-full bg-accent text-accent-foreground text-sm font-bold hover:opacity-90 transition-opacity"
            >
              جرّب المساعد الآن
              <ArrowLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => go("library")}
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-secondary text-secondary-foreground font-bold text-sm hover:bg-secondary/70 transition-colors"
            >
              <BookOpen className="w-4 h-4" />
              استعرض المكتبة
              ابحث
            </button>
          </div>

          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-primary" /> إجابات موثّقة بالمصدر
            </span>
          <div className="mt-7 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
            <span>+٤٠٠ مرجع أكاديمي</span>
            <span>دعم كامل للعربية والإنجليزية</span>
            <span>إجابات موثّقة بالمصدر</span>
            <span>عربي وإنجليزي</span>
          </div>
        </motion.div>
          className="relative"
        >
          <div className="rounded-3xl overflow-hidden border border-border shadow-soft">
            <img
              src={heroImg}
              alt="شبكة من الأفراد والكتب ترمز إلى المعرفة في الخدمة الاجتماعية وعلم الاجتماع"
              width={1536}
              height={1024}
              className="w-full h-auto"
            />
          </div>
          <div className="absolute -bottom-6 right-6 surface-card px-5 py-4 max-w-[15rem] hidden sm:block">
            <p className="text-xs text-muted-foreground">دقة الإجابات المرجعية</p>
            <p className="text-2xl font-extrabold text-primary mt-1">٩٦٪</p>
          </div>
          <img
            src={heroImg}
            alt="يد تحمل مجموعة كتب في الخدمة الاجتماعية وعلم الاجتماع"
            width={1200}
            height={1200}
            className="w-full h-auto max-w-md mx-auto"
          />

          <span className="float-chip bg-secondary top-6 right-2 sm:right-6">
            <Lightbulb className="w-6 h-6 text-primary" />
          </span>
          <span className="float-chip bg-card top-20 left-2 sm:left-4">
            <Award className="w-6 h-6 text-accent" />
          </span>
          <span className="float-chip bg-secondary bottom-16 left-6 sm:left-12">
            <Target className="w-6 h-6 text-primary" />
          </span>
        </motion.div>
      </div>
