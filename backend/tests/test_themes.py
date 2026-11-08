"""主题系统完整性：style.css 引用的 --han-* 颜色变量必须在默认作用域定义；
index.html 引用的主题 CSS 必须存在；当前启用的主题（landscape）必须定义核心颜色集。"""
import os
import re
import unittest

FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend")


def _read(name):
    with open(os.path.join(FRONTEND, name), encoding="utf-8") as f:
        return f.read()


class TestThemeIntegrity(unittest.TestCase):
    style = _read("style.css")
    themes = _read("themes-scoped.css")

    def test_used_vars_have_fallback_definition(self):
        used = set(re.findall(r"var\((--han-[a-z0-9-]+)", self.style))
        defined = set(re.findall(r"^\s*(--han-[a-z0-9-]+)\s*:", self.style, re.M))
        self.assertEqual(used - defined, set(), "使用了未定义的变量")

    def test_active_theme_defines_core_colors(self):
        m = re.search(r'data-theme="(\w+)"', _read("index.html"))
        active = m.group(1) if m else None
        if not active:
            self.skipTest("未启用具名主题")
        used = set(re.findall(r"var\((--han-[a-z0-9-]+)", self.style))
        style_defined = set(re.findall(r"^\s*(--han-[a-z0-9-]+)\s*:", self.style, re.M))
        block = re.search(
            r":where\(\[data-han-scope\]\[data-theme='%s'\][^)]*\)\s*\{(.*?)\n\}" % active,
            self.themes, re.S)
        if not block:
            self.skipTest("主题 %s 无覆盖块（沿用默认）" % active)
        defined = set(re.findall(r"^\s*(--han-[a-z0-9-]+)\s*:", block.group(1), re.M))
        colorish = {v for v in used & style_defined
                    if re.search(r"--han-(color|seal|stamp|nav|scroll|focus-ring)", v)}
        missing = colorish - defined
        self.assertEqual(missing, set(), "主题 %s 缺少颜色变量: %s" % (active, missing))

    def test_referenced_css_files_exist(self):
        html = _read("index.html")
        for href in re.findall(r'stylesheet"\s+href="/([^"]+)"', html):
            self.assertTrue(os.path.isfile(os.path.join(FRONTEND, href)),
                            "index.html 引用了不存在的 %s" % href)


if __name__ == "__main__":
    unittest.main()
