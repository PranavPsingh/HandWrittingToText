from kivy.app import App
from kivy.uix.label import Label


class HandwritingHelperApp(App):
    def build(self) -> Label:
        self.title = "Handwriting Helper"
        return Label(text="Kivy works. Handwriting Helper is alive.")


if __name__ == "__main__":
    HandwritingHelperApp().run()
    