import fiftyone.operators as foo
from fiftyone.operators.types import View, Object, Choices, Property, GridView, TableView, Object as TypeObject
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix


class DecomposeCorePanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="decompose_core_panel",
            label="Concept Decomposition",
            description="Show SpLiCE decomposition for image and dataset",
            dynamic=True,
        )
    def load_panel_data(self, ctx):
        if hasattr(self, "_cache"):
            return self._cache
        
        label_field = ctx.params.get("label_field", "concepts")
        top_k = ctx.params.get("top_k", 20)
        dataset = ctx.dataset

        ids = dataset.values("id")
        id_2_index = {sid: idx for idx, sid in enumerate(ids)}
        num_samples = len(ids)

        labels_array = np.array(
            dataset.values(f"{label_field}.classifications.label")
        )

        weights_array = np.array(
            dataset.values(f"{label_field}.classifications.weight"),
            dtype=np.float32,
        )

        vocabulary = np.unique(labels_array[labels_array != None])
        col_indices = np.searchsorted(vocabulary, labels_array)

        valid_mask = labels_array != None
        row_indices = np.repeat(np.arange(num_samples), labels_array.shape[1])
        row_indices = row_indices[valid_mask.flatten()]
        col_indices_flat = col_indices.flatten()[valid_mask.flatten()]
        weights_flat = weights_array.flatten()[valid_mask.flatten()]

        concept_array = csr_matrix(
            (weights_flat, (row_indices, col_indices_flat)),
            shape=(num_samples, len(vocabulary)),
        )

        self._cache = {
            "concept_array": concept_array,
            "vocabulary": vocabulary,
            "id_2_index": id_2_index,
            "top_k": top_k,
            "l0_norms": np.array(
                dataset.values(f"{label_field}.l0_norm"),
                dtype=np.float32,
            ),
            "cosine_sims": np.array(
                dataset.values(f"{label_field}.reconstruction_error"),
                dtype=np.float32,
            ),
            "all_class_labels": np.array(
                dataset.values("ground_truth.detections.label"),
                dtype=object,
            ),
        }

    def on_load(self, ctx, init=True):
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

        self.load_panel_data(ctx)
        concept_array = self._cache["concept_array"]
        vocabulary = self._cache["vocabulary"]
        id_2_index = self._cache["id_2_index"]
        top_k = self._cache["top_k"]
        all_class_labels = self._cache["all_class_labels"]
        l0_norms = self._cache.get("l0_norms", None)
        cosine_sims = self._cache.get("cosine_sims", None)

        # Check if selected or view
        ids = ctx.selected if ctx.selected else ctx.view.values("id")
        analysis_source = "Selected Samples" if ctx.selected else "Current View"
        indices = [id_2_index[_id] for _id in ids if _id in id_2_index]

        # Subset concept matrix directly using indices
        subset = concept_array[indices]

        if page == 1:
            mean_weights = subset.mean(axis=0).A1
            counts = (subset > 0).sum(axis=0).A1

            mask = counts > 0
            i = np.argsort(-mean_weights[mask])[:top_k]
            v, w, n = vocabulary[mask][i], mean_weights[mask][i], counts[mask][i]

            ctx.panel.state.set(
                "dataset_table",
                [dict(concept=c, mean_weight=float(m), count=int(k)) for c, m, k in zip(v, w, n)]
            )
            ctx.panel.state.set(
                "dataset_plot",
                dict(x=w.tolist(), y=v.tolist(), type="bar", orientation="h")
            )

            subset_l0 = l0_norms[indices] if l0_norms is not None else None
            subset_cos = cosine_sims[indices] if cosine_sims is not None else None
            avg_l0 = np.nanmean(subset_l0) if subset_l0 is not None and subset_l0.size else 0.0
            avg_cos = np.nanmean(subset_cos) if subset_cos is not None and subset_cos.size else 0.0

            ctx.panel.state.set(
                "dataset_info_md",
                f"""
                ###### 📉 Decomposition Statistics ({analysis_source})
                | Metric | Value |
                |--------|-------|
                | **Avg. Decomposition L0 Norm** | `{avg_l0:.4f}` |
                | **Avg. CLIP–SpLiCE Cosine Similarity** | `{avg_cos:.4f}` |
                | **Samples Analyzed** | `{subset.shape[0]}` |
                """
            )

        elif page == 2:
            # Page 2: Spurious correlation analysis
            labels_map = {idx: all_class_labels[idx] for idx in indices}

            # Initialize structures
            concept_hist = defaultdict(lambda: defaultdict(int))
            
            # Use COO representation for efficient iteration
            coo = subset.tocoo()
            concepts = vocabulary[coo.col]

            for row_idx, concept in zip(coo.row, concepts):
                global_idx = indices[row_idx]
                classes = labels_map.get(global_idx)
                if not classes:
                    continue

                for cls in classes:
                    concept_hist[concept][cls] += 1

            concept_choices = sorted(concept_hist.keys())
            ctx.panel.state.set("concept_choices", concept_choices)

            selected_concept = ctx.panel.get_state(
                "selected_concept",
                concept_choices[0] if concept_choices else None,
            )
            ctx.panel.state.set("selected_concept", selected_concept)

            if selected_concept:
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

                PDT: Concept weights are aggregated per sample before computing mean weights across samples.
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


##
