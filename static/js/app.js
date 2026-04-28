const revealItems = document.querySelectorAll(".reveal");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  { threshold: 0.12 }
);

revealItems.forEach((item) => observer.observe(item));

const fillButton = document.getElementById("fill-median");

if (fillButton) {
  fillButton.addEventListener("click", () => {
    document.querySelectorAll(".input-card input").forEach((input) => {
      const fallback = input.dataset.default;
      if (!fallback) {
        return;
      }
      input.value = fallback;
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  });
}

window.addEventListener("load", () => {
  setTimeout(() => {
    document.querySelectorAll(".reveal").forEach((el) => el.classList.add("visible"));
  }, 100);
});
