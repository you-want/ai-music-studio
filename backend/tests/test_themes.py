"""主题系统完整性：style.css 引用的 --han-* 变量必须同时定义于
默认作用域与 themes-scoped.css 的每一个主题块；主题入口只导入存在的文件。"""
import os
import re
import unittest

FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend")


def _read(name):
    with open(os.path.join(FRONTEND, name), encoding="utf-8") as f:
        return f.read()


class TestThemeCoverage(unittest.TestCase):
    style = _read("style.css")
    themes = _read("themes-scoped.css")
    entry = _read("han-scoped.css")

    def test_used_vars_have_fallback_definition(self):
        """style.css 用到的每个 --han-* 变量必须在默认作用域有定义。"""
        used = set(re.findall(r"var\((--han-[a-z0-9-]+)", self.style))
        defined = set(re.findall(r"^\s*(--han-[a-z0-9-]+)\s*:", self.style, re.M))
        missing = used - defined
        self.assertEqual(missing, set(), "style.css 使用了未定义的变量: %s" % missing)

    def test_every_theme_defines_core_color_vars(self):
        """16 套主题的每一套都必须定义核心颜色变量集（结构令牌如圆角/时长
        由默认作用域提供，允许缺省——CSS 级联回落到 [data-han-scope] 默认值）。"""
        used = set(re.findall(r"var\((--han-[a-z0-9-]+)", self.style))
        parts = re.split(r":where\(\[data-han-scope\]\[data-theme='(\w+)'", self.themes)
        themes = dict(zip(parts[1::2], parts[2::2]))
        self.assertGreaterEqual(len(themes), 16, "主题数量不应减少（当前 %d）" % len(themes))
        # 核心集 = 使用中、默认作用域已定义、且属于“随主题变化的颜色类”变量
        # （字体/圆角/时长/缓动等结构令牌由默认作用域提供，主题无需重复定义）
        style_defined = set(re.findall(r"^\s*(--han-[a-z0-9-]+)\s*:", self.style, re.M))
        colorish = re.compile(r"--han-(color|seal|stamp|nav|scroll|focus-ring)")
        core = {v for v in used & style_defined if colorish.search(v)}
        self.assertTrue(core, "应存在随主题变化的颜色变量")
        for name, body in themes.items():
            defined = set(re.findall(r"^\s*(--han-[a-z0-9-]+)\s*:", body, re.M))
            missing = core - defined
            self.assertEqual(missing, set(), "主题 %s 缺少核心变量: %s" % (name, missing))

    def test_entry_imports_only_existing_files(self):
        """han-scoped.css 的 @import 不能指向不存在的文件（避免 404 请求）。"""
        for target in re.findall(r'@import url\("\./([^"]+)"\)', self.entry):
            self.assertTrue(os.path.isfile(os.path.join(FRONTEND, target)),
                            "han-scoped.css 导入了不存在的 %s" % target)

    def test_switcher_lists_all_themes(self):
        """前端切换器的主题清单必须覆盖 themes-scoped.css 的全部主题。"""
        js = open(os.path.join(FRONTEND, "app.js"), encoding="utf-8").read()
        listed = set(re.findall(r'id:\s*"(\w+)"', js))
        parts = re.split(r":where\(\[data-han-scope\]\[data-theme='(\w+)'", self.themes)
        for name in parts[1::2]:
            self.assertIn(name, listed, "切换器缺少主题 %s" % name)


if __name__ == "__main__":
    unittest.main()
