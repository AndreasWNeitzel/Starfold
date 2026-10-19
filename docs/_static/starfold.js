// Mark the "Formula" dropdowns so the stylesheet can give them the highlight rule.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("details.sd-dropdown").forEach(function (box) {
    var title = box.querySelector(".sd-summary-text") || box.querySelector("summary");
    if (title && title.textContent.trim() === "Formula") {
      box.classList.add("sf-formula");
    }
  });
});
