import fiftyone.operators as foo
from fiftyone.operators.types import View, Object, Choices, Property, GridView, TableView, Object as TypeObject
from collections import defaultdict


class DecomposeCorePanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="decompose_core_panel",
            label="Concept Decomposition",
            description="Show SpLiCE decomposition for image and dataset",
            dynamic=True,
        )

    def on_load(self, ctx, init=False):
        ctx.panel.state.set("page", 1)
        self._update(ctx)

    def on_class_change(self, ctx):
        self._update(ctx)

    def go_to_page_4(self, ctx):
        ctx.panel.state.set("page", 4)
        self._update(ctx)

    def go_to_page_3(self, ctx):
        ctx.panel.state.set("page", 3)
        self._update(ctx)

    def go_to_page_2(self, ctx):
        ctx.panel.state.set("page", 2)
        self._update(ctx)

    def go_to_page_1(self, ctx):
        ctx.panel.state.set("page", 1)
        self._update(ctx)

    def _update(self, ctx):
        page = ctx.panel.get_state("page", 1)

        if page == 1:
            concept_agg = defaultdict(list)
            total_samples = 0
            l0_norms = []
            cosine_sims = []

            for sample in ctx.dataset:
                concepts = getattr(sample, "splice_concepts", None)
                if not concepts:
                    continue

                total_samples += 1
                for concept in concepts:
                    concept_agg[concept["concept"]].append(concept["weight"])

                l0_norms.append(getattr(sample, "splice_l0_norm", 0))
                cosine_sims.append(getattr(sample, "splice_cosine_sim", 0))

            concept_stats = [
                {
                    "concept": name,
                    "mean_weight": sum(weights) / len(weights),
                    "count": len(weights)
                }
                for name, weights in concept_agg.items()
            ]
            concept_stats = sorted(concept_stats, key=lambda x: x["mean_weight"], reverse=True)[:20]
            ctx.panel.state.set("dataset_table", concept_stats)

            ctx.panel.state.set(
                "dataset_plot",
                {
                    "x": [c["mean_weight"] for c in concept_stats],
                    "y": [c["concept"] for c in concept_stats],
                    "type": "bar",
                    "orientation": "h",
                },
            )

            ctx.panel.state.set(
                "dataset_info_md",
                f"""
                ###### 📉 Decomposition Statistics
                | Metric | Value |
                |--------|-------|
                | **Avg. Decomposition L0 Norm** | `{sum(l0_norms)/len(l0_norms):.4f}` |
                | **Avg. CLIP–SpLiCE Cosine Similarity** | `{sum(cosine_sims)/len(cosine_sims):.4f}` |
                """
            )

        elif page == 2:
            labels = set()
            for sample in ctx.dataset:
                detections = getattr(sample, "ground_truth", None)
                if detections and detections.detections:
                    for det in detections.detections:
                        labels.add(det.label)

            label_choices = sorted(labels)
            ctx.panel.state.set("class_choices", label_choices)

            selected_class = ctx.panel.get_state("selected_class", label_choices[0] if label_choices else None)
            ctx.panel.state.set("selected_class", selected_class)

            concept_agg = defaultdict(list)
            l0_norms = []
            cosine_sims = []

            for sample in ctx.dataset:
                detections = getattr(sample, "ground_truth", None)
                if not detections or not detections.detections:
                    continue

                labels_in_sample = [det.label for det in detections.detections]
                if selected_class not in labels_in_sample:
                    continue

                concepts = getattr(sample, "splice_concepts", None)
                if not concepts:
                    continue

                for concept in concepts:
                    concept_agg[concept["concept"]].append(concept["weight"])

                l0_norms.append(getattr(sample, "splice_l0_norm", 0))
                cosine_sims.append(getattr(sample, "splice_cosine_sim", 0))

            concept_stats = [
                {
                    "concept": name,
                    "mean_weight": sum(weights) / len(weights),
                    "count": len(weights),
                }
                for name, weights in concept_agg.items()
            ]
            concept_stats = sorted(concept_stats, key=lambda x: x["mean_weight"], reverse=True)[:20]
            ctx.panel.state.set("class_table", concept_stats)

            ctx.panel.state.set(
                "class_plot",
                {
                    "x": [c["mean_weight"] for c in concept_stats],
                    "y": [c["concept"] for c in concept_stats],
                    "type": "bar",
                    "orientation": "h",
                },
            )

            if l0_norms:
                ctx.panel.state.set(
                    "class_info_md",
                    f"""
                    ###### 📊 Class Decomposition Summary for `{selected_class}`

                    | Metric | Value |
                    |--------|-------|
                    | **Avg. Decomposition L0 Norm** | `{sum(l0_norms)/len(l0_norms):.4f}` |
                    | **Avg. CLIP–SpLiCE Cosine Similarity** | `{sum(cosine_sims)/len(cosine_sims):.4f}` |
                    """
                )
        elif page == 3:
            sel = ctx.selected
            if not sel:
                ctx.panel.state.set("concept_data", None)
                ctx.panel.state.set("image_info_md", "⚠️ No sample selected.")
                return

            sample = ctx.dataset[sel[0]]
            concepts = getattr(sample, "splice_concepts", None)
            if not concepts:
                ctx.panel.state.set("concept_data", None)
                ctx.panel.state.set("image_info_md", "⚠️ No concept decomposition found.")
                return

            x = [c["weight"] for c in concepts]
            y = [c["concept"] for c in concepts]

            ctx.panel.state.set(
                "concept_data",
                {
                    "x": x,
                    "y": y,
                    "type": "bar",
                    "orientation": "h",
                },
            )

            l0 = getattr(sample, "splice_l0_norm", 0)
            cos = getattr(sample, "splice_cosine_sim", 0)
            filename = sample.filepath.split("/")[-1]

            concept_table = [
                {"concept": c["concept"], "mean_weight": c["weight"], "count": 1}
                for c in concepts
            ]
            ctx.panel.state.set("image_table", concept_table)

            ctx.panel.state.set(
                "image_info_md",
                f"""
                ###### 🖼️ Decomposition Info for `{filename}`

                | Metric | Value |
                |--------|-------|
                | **Decomposition L0 Norm** | `{l0:.4f}` |
                | **CLIP–SpLiCE Cosine Similarity** | `{cos:.4f}` |
                """
            )
        elif page == 4:
            concept_hist = defaultdict(lambda: defaultdict(int))
            all_concepts = set()
            all_labels = set()

            for sample in ctx.dataset:
                concepts = getattr(sample, "splice_concepts", None)
                detections = getattr(sample, "ground_truth", None)

                if not concepts or not detections or not detections.detections:
                    continue

                labels = {det.label for det in detections.detections}
                all_labels.update(labels)

                for concept in concepts:
                    all_concepts.add(concept["concept"])
                    for label in labels:
                        concept_hist[concept["concept"]][label] += 1

            concept_choices = sorted(all_concepts)
            ctx.panel.state.set("concept_choices", concept_choices)

            selected_concept = ctx.panel.get_state("selected_concept", concept_choices[0])
            ctx.panel.state.set("selected_concept", selected_concept)

            data = concept_hist[selected_concept]
            ctx.panel.state.set(
                "spurious_plot",
                {
                    "x": list(data.keys()),
                    "y": list(data.values()),
                    "type": "bar",
                }
            )

    def render(self, ctx):
        view = GridView(align_x="center", align_y="center", orientation="vertical", height=100, width=100, gap=2)
        panel = Object()

        page = ctx.panel.get_state("page", 1)

        if page == 1:
            table = TableView()
            table.add_column("concept", label="Concept")
            table.add_column("mean_weight", label="Mean Weight")
            table.add_column("count", label="Count")

            panel.md("&nbsp;", name="spacer")
            panel.md(
                """
 
                #### 🧬 Dataset-Level Concept Summary 📊

                Welcome to the **SpLiCE Global View**! This page shows an overview of the most significant concepts across your entire dataset.
                
                &nbsp;

                #####  📌 What You See:
                - 🧠 **Top 10 Concepts** ranked by their average contribution
                - 📈 **Mean Weight**: how influential each concept is across all samples
                - 🔢 **Count**: how many samples each concept appeared in

                """, 
                name="dataset_intro"
            )

            panel.list("dataset_table", TypeObject(), view=table, label="Top Concepts")
            panel.plot(
                "dataset_plot",
                layout={
                    "title": {"text": "Top Concept Distribution", "xanchor": "center"},
                    "xaxis": {"title": "Mean Weight"},
                    "yaxis": {"title": "Concept", "autorange": "reversed"},
                    "margin": {"l": 150, "t": 60},
                },
                width=100,
            )
            
            panel.md(ctx.panel.state.get("dataset_info_md", ""), name="dataset_stats", label=None)
            panel.md("&nbsp;", name="spacer")

        elif page == 2:
            panel.md("&nbsp;", name="spacer")
            panel.md("#### 🏷️ Class-Level Decomposition", name="class_title")

            class_choices = ctx.panel.state.get("class_choices", [])
            panel.enum(
                "selected_class",
                class_choices,
                default=class_choices[0] if class_choices else None,
                on_change=self.on_class_change,
                label="Select Class Label"
            )

            table = TableView()
            table.add_column("concept", label="Concept")
            table.add_column("mean_weight", label="Mean Weight")
            table.add_column("count", label="Count")

            panel.list("class_table", TypeObject(), view=table, label="Top Concepts for Class")

            panel.plot(
                "class_plot",
                layout={
                    "title": {"text": f"Top Concept Distribution for {ctx.panel.state.get('selected_class', '')}", "xanchor": "center"},
                    "xaxis": {"title": "Mean Weight"},
                    "yaxis": {"title": "Concept", "autorange": "reversed"},
                    "margin": {"l": 150, "t": 60},
                },
                width=100,
            )
            
            panel.md(ctx.panel.state.get("class_info_md", ""), name="class_stats", label=None)
            panel.md("&nbsp;", name="spacer")
        elif page == 3:
            panel.md("&nbsp;", name="spacer")
            panel.md("#### 🖼️ Image-Level Decomposition", name="image_title")
            panel.btn("refresh_image", label="🔄 Refresh to see new image", on_click=self._update)

            table = TableView()
            table.add_column("concept", label="Concept")
            table.add_column("mean_weight", label="Weight")
            table.add_column("count", label="Count")

            panel.list("image_table", TypeObject(), view=table, label="Concept Decomposition Table")

            panel.plot(
                "concept_data",
                layout={
                    "title": {"text": "Concept Decomposition for Selected Image", "xanchor": "center"},
                    "xaxis": {"title": "Weight"},
                    "yaxis": {"title": "Concept", "autorange": "reversed"},
                    "margin": {"l": 150, "t": 60},
                },
                width=100,
            )

            panel.md(ctx.panel.state.get("image_info_md", ""), name="image_stats", label=None)
            panel.md("&nbsp;", name="spacer")

        elif page == 4:
            panel.md(
                """
                &nbsp;
                #### 🧪 Spurious Correlation Discovery

                This view helps uncover **unintended correlations** between concepts and class labels. It is especially useful for identifying **biases** in your dataset or model.
                
                &nbsp;

                ##### 🔍 What You See:
                - 📌 **Select a Concept** from the dropdown below.
                - 📊 The bar chart shows how often this concept appears in each **class**.
                - ⚠️ If a concept occurs mostly in a single class, it may be **spuriously correlated** with that label — even if it's not semantically relevant.

                ##### 💡 Why It Matters:
                Spurious correlations can cause models to make decisions based on irrelevant cues (e.g., background, lighting, texture).  
                Use this to find and fix dataset shortcuts or model failures.

                """,
                name="spurious_intro"
            )
            
            concept_choices = ctx.panel.state.get("concept_choices", [])
            panel.enum(
                "selected_concept",
                concept_choices,
                default=concept_choices[0] if concept_choices else None,
                on_change=self._update,
                label="Select Concept",
            )

            panel.plot(
                "spurious_plot",
                layout={
                    "title": {
                        "text": "Concept Occurrence Across Classes",
                        "xanchor": "center"
                    },
                    "xaxis": {"title": "Class"},
                    "yaxis": {"title": "Count"},
                    "margin": {"t": 80}  
                },
                width=100,
            )

        panel.arrow_nav(
            "page_nav",
            forward=page < 4,
            backward=page > 1,
            on_forward=self.go_to_page_2 if page == 1 else self.go_to_page_3 if page == 2 else self.go_to_page_4,
            on_backward=self.go_to_page_1 if page == 2 else self.go_to_page_2 if page == 3 else self.go_to_page_3,
        )

        return Property(panel, view=view)


def register(p):
    p.register(DecomposeCorePanel)