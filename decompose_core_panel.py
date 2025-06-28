import fiftyone.operators as foo
from fiftyone.operators.types import View, Object, Choices, Property, GridView, TableView, Object as TypeObject
from collections import defaultdict
import numpy as np

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
    
    def on_change_ctx(self, ctx):
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

        sel = ctx.selected
        if sel:
            samples_to_analyze = [ctx.view[sample_id] for sample_id in sel]
            analysis_source = "Selected Samples"
        else:
            samples_to_analyze = ctx.view
            analysis_source = "Current View"

        if page == 1:
            concept_agg = defaultdict(list)
            total_samples = 0
            all_l0_norms = []
            all_cosine_sims = []

            for sample in samples_to_analyze:
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
                ###### 📉 Decomposition Statistics ({analysis_source})
                | Metric | Value |
                |--------|-------|
                | **Avg. Decomposition L0 Norm** | `{avg_l0:.4f}` |
                | **Avg. CLIP–SpLiCE Cosine Similarity** | `{avg_cos:.4f}` |
                | **Samples Analyzed** | `{total_samples}` |
                """
            )
        elif page == 2:
            # Spurious correlation analysis
            concept_hist = defaultdict(lambda: defaultdict(int))
            all_concepts = set()
            all_classes = set()

            for sample in samples_to_analyze:
                detections = getattr(sample, "ground_truth", None)
                if not detections or not detections.detections:
                    continue
                
                sample_labels = [det.label for det in detections.detections]
                all_classes.update(sample_labels)
                
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
 
                #### 🧬 Comprehensive Concept Analysis 📊

                Welcome to the **SpLiCE Unified View**! This page combines dataset-level, class-level, and image-level concept analysis in one comprehensive interface.
                
                &nbsp;

                #####  📌 What You Can Do Here:
                - 🧠 **Dataset Overview**: See top concepts across your current view with dynamic updates
                - 🏷️ **Class Analysis**: Select any class from the dropdown below to see associated concepts
                - 🖼️ **Image Details**: Select any image in FiftyOne to see its individual concept decomposition
                - 📈 **Dynamic Updates**: All statistics automatically update based on your current view and selections

                #####  🔄 How to Use:
                1. **Filter your dataset** using FiftyOne's view controls to analyze specific subsets
                2. **Select a class** from the dropdown to see concepts for that specific category
                3. **Click on any image** in the FiftyOne grid to see its detailed concept breakdown
                4. **Navigate to Page 2** for spurious correlation analysis

                """, 
                name="unified_intro"
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
            forward=page < 2,
            backward=page > 1,
            on_forward=self.go_to_page_2 if page == 1 else None,
            on_backward=self.go_to_page_1 if page == 2 else None,
        )

        return Property(panel, view=view)


def register(p):
    p.register(DecomposeCorePanel)