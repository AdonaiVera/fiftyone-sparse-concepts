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
        label_field = ctx.params.get("label_field", "concepts")

        if page == 1:
            # Dataset-level aggregation
            concept_agg = defaultdict(list)
            total_samples = 0
            all_l0_norms = []
            all_cosine_sims = []

            for sample in ctx.dataset:
                labels_obj = sample[label_field] if label_field in sample else None
                if not labels_obj or not labels_obj.classifications:
                    continue
                
                total_samples += 1
                for classification in labels_obj.classifications:
                    concept_agg[classification.label].append(classification.weight)
                
                if hasattr(labels_obj, 'l0_norm') and labels_obj.l0_norm is not None:
                    all_l0_norms.append(labels_obj.l0_norm)
                if hasattr(labels_obj, 'reconstruction_error') and labels_obj.reconstruction_error is not None:
                    all_cosine_sims.append(labels_obj.reconstruction_error)

            concept_stats = [
                {
                    "concept": name,
                    "mean_weight": sum(weights) / len(weights),
                    "count": len(weights),
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

            avg_l0 = sum(all_l0_norms) / len(all_l0_norms) if all_l0_norms else 0
            avg_cos = sum(all_cosine_sims) / len(all_cosine_sims) if all_cosine_sims else 0
            
            ctx.panel.state.set(
                "dataset_info_md",
                f"""
                ###### 📉 Decomposition Statistics
                | Metric | Value |
                |--------|-------|
                | **Avg. Decomposition L0 Norm** | `{avg_l0:.4f}` |
                | **Avg. CLIP–SpLiCE Cosine Similarity** | `{avg_cos:.4f}` |
                """
            )

        elif page == 2:
            # Class-level aggregation
            all_detections = ctx.dataset.values("ground_truth.detections.label")

            labels = set()
            for dets in all_detections:
                if dets:
                    labels.update(dets)

            label_choices = sorted(labels)
            ctx.panel.state.set("class_choices", label_choices)

            selected_class = ctx.panel.get_state("selected_class", label_choices[0] if label_choices else None)
            ctx.panel.state.set("selected_class", selected_class)

            concept_agg = defaultdict(list)
            filtered_l0 = []
            filtered_cos = []

            for sample in ctx.dataset:
                # Check if this sample has the selected class
                detections = getattr(sample, "ground_truth", None)
                if not detections or not detections.detections:
                    continue
                
                sample_labels = [det.label for det in detections.detections]
                if selected_class not in sample_labels:
                    continue
                
                # Get concept classifications
                labels_obj = sample[label_field] if label_field in sample else None
                if not labels_obj or not labels_obj.classifications:
                    continue
                
                for classification in labels_obj.classifications:
                    concept_agg[classification.label].append(classification.weight)
                
                if hasattr(labels_obj, 'l0_norm') and labels_obj.l0_norm is not None:
                    filtered_l0.append(labels_obj.l0_norm)
                if hasattr(labels_obj, 'reconstruction_error') and labels_obj.reconstruction_error is not None:
                    filtered_cos.append(labels_obj.reconstruction_error)

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

            if filtered_l0:
                avg_l0 = sum(filtered_l0) / len(filtered_l0)
                avg_cos = sum(filtered_cos) / len(filtered_cos) if filtered_cos else 0
                ctx.panel.state.set(
                    "class_info_md",
                    f"""
                    ###### 📊 Class Decomposition Summary for `{selected_class}`

                    | Metric | Value |
                    |--------|-------|
                    | **Avg. Decomposition L0 Norm** | `{avg_l0:.4f}` |
                    | **Avg. CLIP–SpLiCE Cosine Similarity** | `{avg_cos:.4f}` |
                    """
                )
        elif page == 3:
            sel = ctx.selected
            if not sel:
                ctx.panel.state.set("concept_data", None)
                ctx.panel.state.set("image_info_md", "⚠️ No sample selected.")
                return

            sample = ctx.dataset[sel[0]]
            labels_obj = sample[label_field] if label_field in sample else None
            if not labels_obj or not labels_obj.classifications:
                ctx.panel.state.set("concept_data", None)
                ctx.panel.state.set("image_info_md", "⚠️ No concept decomposition found.")
                return

            x = [c.weight for c in labels_obj.classifications]
            y = [c.label for c in labels_obj.classifications]

            concept_table = [
                {"concept": c.label, "mean_weight": c.weight, "count": 1}
                for c in labels_obj.classifications
            ]

            l0 = getattr(labels_obj, "l0_norm", 0)
            cos = getattr(labels_obj, "reconstruction_error", 0)
            filename = sample.filepath.split("/")[-1]

            ctx.panel.state.set(
                "concept_data",
                {
                    "x": x,
                    "y": y,
                    "type": "bar",
                    "orientation": "h",
                },
            )
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
            # Spurious correlation analysis
            concept_hist = defaultdict(lambda: defaultdict(int))
            all_concepts = set()
            all_classes = set()

            for sample in ctx.dataset:
                # Get detections
                detections = getattr(sample, "ground_truth", None)
                if not detections or not detections.detections:
                    continue
                
                sample_labels = [det.label for det in detections.detections]
                all_classes.update(sample_labels)
                
                # Get concept classifications
                labels_obj = sample[label_field] if label_field in sample else None
                if not labels_obj or not labels_obj.classifications:
                    continue
                
                for classification in labels_obj.classifications:
                    all_concepts.add(classification.label)
                    for cls in sample_labels:
                        concept_hist[classification.label][cls] += 1

            concept_choices = sorted(all_concepts)
            ctx.panel.state.set("concept_choices", concept_choices)

            selected_concept = ctx.panel.get_state("selected_concept", concept_choices[0] if concept_choices else None)
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