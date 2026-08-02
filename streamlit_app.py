import { motion } from "framer-motion";
import { Award, Lightbulb, Search, Target } from "lucide-react";
import heroImg from "@/assets/hero-books.jpg";

const Hero = () => {
  const go = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });

  return (
    <section className="relative pt-28 sm:pt-32 pb-16 overflow-hidden">
      <div className="max-w-6xl mx-auto px-5 grid lg:grid-cols-2 gap-10 lg:gap-6 items-center">
        <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <h1 className="text-4xl sm:text-5xl lg:text-[3.6rem] font-extrabold leading-[1.2] text-foreground">
            اعثر على الإجابة
            <span className="block">التي تبحث عنها في</span>
            <span className="block text-primary">الخدمة الاجتماعية.</span>
          </h1>
          <p className="mt-5 text-sm sm:text-base text-muted-foreground leading-relaxed max-w-md">
            مساعد ذكي يجيبك من كتب ومراجع الخدمة الاجتماعية وعلم الاجتماع، مع ذكر المصدر ورقم الصفحة.
          </p>

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
              className="shrink-0 px-6 py-3 rounded-full bg-accent text-accent-foreground text-sm font-bold hover:opacity-90 transition-opacity"
            >
              ابحث
            </button>
          </div>

          <div className="mt-7 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
            <span>+٤٠٠ مرجع أكاديمي</span>
            <span>إجابات موثّقة بالمصدر</span>
            <span>عربي وإنجليزي</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="relative"
        >
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
    </section>
  );
};

export default Hero;
