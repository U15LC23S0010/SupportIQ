from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONVERSATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_annotation_conversations_360_repaired.csv"
)

MESSAGE_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_annotation_messages_360_repaired.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "tesco_golden_human_annotations_working.csv"
)

TARGET_INTENTS = [
    "delivery_order",
    "pricing_payment",
    "store_staff_service",
    "product_availability",
    "product_quality",
    "product_safety_sensitive",
    "product_information_policy",
    "feedback_suggestion",
]

REVIEW_ONLY = [
    "other_unclear",
    "non_support_social",
]

CONFIDENCE_OPTIONS = [
    "high",
    "medium",
    "low",
]

YES_NO = [
    "yes",
    "no",
]


def ensure_output_file() -> None:
    """
    Create the working annotation file if it does not already exist.

    This file is NOT the final golden dataset.
    """

    if OUTPUT_PATH.exists():
        return

    if not CONVERSATION_PATH.exists():
        raise FileNotFoundError(
            f"Conversation file not found:\n{CONVERSATION_PATH}"
        )

    df = pd.read_csv(
        CONVERSATION_PATH,
        keep_default_na=False,
    )

    defaults = {
        "evaluation_incoming_tweet_id": "",
        "evaluation_incoming_message": "",
        "human_intent": "",
        "annotator_confidence": "",
        "is_multi_intent": "",
        "secondary_issue_present": "",
        "annotation_notes": "",
        "annotation_status": "not_annotated",
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CONVERSATION_PATH.exists():
        raise FileNotFoundError(
            f"Conversation file not found:\n{CONVERSATION_PATH}"
        )

    if not MESSAGE_PATH.exists():
        raise FileNotFoundError(
            f"Message file not found:\n{MESSAGE_PATH}"
        )

    ensure_output_file()

    conversations = pd.read_csv(
        OUTPUT_PATH,
        keep_default_na=False,
    )

    messages = pd.read_csv(
        MESSAGE_PATH,
        keep_default_na=False,
    )

    conversations["conversation_id"] = (
        conversations["conversation_id"]
        .astype(int)
    )

    messages["conversation_id"] = (
        messages["conversation_id"]
        .astype(int)
    )

    messages["tweet_id"] = (
        messages["tweet_id"]
        .astype(int)
    )

    return conversations, messages


class AnnotationTool:
    def __init__(
        self,
        root: tk.Tk,
        conversations: pd.DataFrame,
        messages: pd.DataFrame,
    ) -> None:
        self.root = root
        self.conversations = conversations
        self.messages = messages

        self.current_index = 0

        self.root.title(
            "SupportIQ â€” Golden Annotation Tool"
        )

        self.root.geometry(
            "1150x720"
        )

        self.root.minsize(
            900,
            600,
        )

        self.build_ui()

        self.load_current_conversation()

    def build_ui(self) -> None:
        header = ttk.Frame(
            self.root,
            padding=(10, 8),
        )

        header.pack(
            side=tk.TOP,
            fill=tk.X,
        )

        self.progress_label = ttk.Label(
            header,
            text="",
            font=("Segoe UI", 11, "bold"),
        )

        self.progress_label.pack(
            side=tk.LEFT
        )

        self.conversation_label = ttk.Label(
            header,
            text="",
        )

        self.conversation_label.pack(
            side=tk.RIGHT
        )

        center = ttk.Frame(
            self.root
        )

        center.pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
        )

        self.canvas = tk.Canvas(
            center,
            highlightthickness=0,
            borderwidth=0,
        )

        self.outer_scrollbar = ttk.Scrollbar(
            center,
            orient=tk.VERTICAL,
            command=self.canvas.yview,
        )

        self.canvas.configure(
            yscrollcommand=self.outer_scrollbar.set
        )

        self.outer_scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y,
        )

        self.canvas.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        self.content = ttk.Frame(
            self.canvas,
            padding=10,
        )

        self.canvas_window = (
            self.canvas.create_window(
                (0, 0),
                window=self.content,
                anchor="nw",
            )
        )

        self.content.bind(
            "<Configure>",
            self.update_scroll_region,
        )

        self.canvas.bind(
            "<Configure>",
            self.resize_content,
        )

        self.canvas.bind_all(
            "<MouseWheel>",
            self.mousewheel,
        )

        screening_frame = ttk.LabelFrame(
            self.content,
            text="Screening information â€” NOT a human label",
            padding=8,
        )

        screening_frame.pack(
            fill=tk.X,
            pady=(0, 8),
        )

        self.screening_text = tk.StringVar()

        ttk.Label(
            screening_frame,
            textvariable=self.screening_text,
            wraplength=1050,
        ).pack(
            anchor=tk.W,
        )

        conversation_frame = ttk.LabelFrame(
            self.content,
            text="Full conversation",
            padding=8,
        )

        conversation_frame.pack(
            fill=tk.X,
            pady=(0, 8),
        )

        conversation_container = ttk.Frame(
            conversation_frame
        )

        conversation_container.pack(
            fill=tk.X,
        )

        self.conversation_text = tk.Text(
            conversation_container,
            height=20,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
        )

        conversation_scrollbar = ttk.Scrollbar(
            conversation_container,
            orient=tk.VERTICAL,
            command=self.conversation_text.yview,
        )

        self.conversation_text.configure(
            yscrollcommand=conversation_scrollbar.set
        )

        self.conversation_text.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )

        conversation_scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y,
        )

        target_frame = ttk.LabelFrame(
            self.content,
            text="Evaluation incoming message",
            padding=8,
        )

        target_frame.pack(
            fill=tk.X,
            pady=(0, 8),
        )

        ttk.Label(
            target_frame,
            text=(
                "Choose the genuine incoming CUSTOMER "
                "message that SupportIQ will be evaluated on."
            ),
        ).pack(
            anchor=tk.W,
        )

        # Tweet ID row
        target_id_row = ttk.Frame(
            target_frame
        )

        target_id_row.pack(
            fill=tk.X,
            pady=(6, 4),
        )

        ttk.Label(
            target_id_row,
            text="Customer tweet ID:",
            width=22,
        ).pack(
            side=tk.LEFT
        )

        self.target_id_var = tk.StringVar()

        ttk.Entry(
            target_id_row,
            textvariable=self.target_id_var,
            width=22,
        ).pack(
            side=tk.LEFT,
            padx=(5, 8),
        )

        ttk.Button(
            target_id_row,
            text="Load exact message",
            command=self.load_exact_customer_message,
        ).pack(
            side=tk.LEFT,
        )

        # Exact message
        ttk.Label(
            target_frame,
            text="Exact customer message:",
        ).pack(
            anchor=tk.W,
            pady=(3, 3),
        )

        self.target_message_text = tk.Text(
            target_frame,
            height=4,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
        )

        self.target_message_text.pack(
            fill=tk.X,
            expand=True,
        )

        labeling_frame = ttk.LabelFrame(
            self.content,
            text="Human annotation",
            padding=8,
        )

        labeling_frame.pack(
            fill=tk.X,
            pady=(0, 10),
        )

        # Human intent
        intent_row = ttk.Frame(
            labeling_frame
        )

        intent_row.pack(
            fill=tk.X,
            pady=3,
        )

        ttk.Label(
            intent_row,
            text="Human intent:",
            width=25,
        ).pack(
            side=tk.LEFT
        )

        self.intent_var = tk.StringVar()

        ttk.Combobox(
            intent_row,
            textvariable=self.intent_var,
            values=(
                TARGET_INTENTS
                + REVIEW_ONLY
            ),
            state="readonly",
            width=42,
        ).pack(
            side=tk.LEFT
        )

        # Confidence
        confidence_row = ttk.Frame(
            labeling_frame
        )

        confidence_row.pack(
            fill=tk.X,
            pady=3,
        )

        ttk.Label(
            confidence_row,
            text="Annotator confidence:",
            width=25,
        ).pack(
            side=tk.LEFT
        )

        self.confidence_var = tk.StringVar()

        ttk.Combobox(
            confidence_row,
            textvariable=self.confidence_var,
            values=CONFIDENCE_OPTIONS,
            state="readonly",
            width=20,
        ).pack(
            side=tk.LEFT
        )

        # Multiple issues
        multi_row = ttk.Frame(
            labeling_frame
        )

        multi_row.pack(
            fill=tk.X,
            pady=3,
        )

        ttk.Label(
            multi_row,
            text="Multiple genuine issues?",
            width=25,
        ).pack(
            side=tk.LEFT
        )

        self.multi_var = tk.StringVar()

        ttk.Combobox(
            multi_row,
            textvariable=self.multi_var,
            values=YES_NO,
            state="readonly",
            width=20,
        ).pack(
            side=tk.LEFT
        )

        # Secondary issue
        secondary_row = ttk.Frame(
            labeling_frame
        )

        secondary_row.pack(
            fill=tk.X,
            pady=3,
        )

        ttk.Label(
            secondary_row,
            text="Secondary issue present?",
            width=25,
        ).pack(
            side=tk.LEFT
        )

        self.secondary_var = tk.StringVar()

        ttk.Combobox(
            secondary_row,
            textvariable=self.secondary_var,
            values=YES_NO,
            state="readonly",
            width=20,
        ).pack(
            side=tk.LEFT
        )

        # Notes
        notes_row = ttk.Frame(
            labeling_frame
        )

        notes_row.pack(
            fill=tk.X,
            pady=3,
        )

        ttk.Label(
            notes_row,
            text="Annotation notes:",
            width=25,
        ).pack(
            side=tk.LEFT,
            anchor=tk.N,
        )

        self.notes_text = tk.Text(
            notes_row,
            height=5,
            wrap=tk.WORD,
            font=("Segoe UI", 9),
        )

        self.notes_text.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )

        navigation = ttk.Frame(
            self.root,
            padding=(10, 8),
            relief=tk.RIDGE,
        )

        navigation.pack(
            side=tk.BOTTOM,
            fill=tk.X,
        )

        self.previous_button = ttk.Button(
            navigation,
            text="â† Previous",
            command=self.previous_conversation,
        )

        self.previous_button.pack(
            side=tk.LEFT,
        )

        self.save_button = ttk.Button(
            navigation,
            text="Save annotation",
            command=self.save_current,
        )

        self.save_button.pack(
            side=tk.LEFT,
            padx=8,
        )

        self.next_button = ttk.Button(
            navigation,
            text="Save & Next â†’",
            command=self.save_and_next,
        )

        self.next_button.pack(
            side=tk.LEFT,
        )

        ttk.Label(
            navigation,
            text="Go to candidate:",
        ).pack(
            side=tk.LEFT,
            padx=(30, 5),
        )

        self.jump_var = tk.StringVar()

        ttk.Entry(
            navigation,
            textvariable=self.jump_var,
            width=8,
        ).pack(
            side=tk.LEFT,
        )

        ttk.Button(
            navigation,
            text="Open",
            command=self.jump_to,
        ).pack(
            side=tk.LEFT,
            padx=5,
        )

        self.status_label = ttk.Label(
            navigation,
            text="",
        )

        self.status_label.pack(
            side=tk.LEFT,
            padx=(20, 0),
        )

        ttk.Button(
            navigation,
            text="Exit",
            command=self.exit_tool,
        ).pack(
            side=tk.RIGHT,
        )

        # Keyboard shortcuts
        self.root.bind(
            "<Control-s>",
            lambda event: self.save_current(),
        )

        self.root.bind(
            "<Right>",
            lambda event: self.save_and_next(),
        )

        self.root.bind(
            "<Left>",
            lambda event: self.previous_conversation(),
        )

    def update_scroll_region(
        self,
        event=None,
    ) -> None:
        self.canvas.configure(
            scrollregion=self.canvas.bbox(
                "all"
            )
        )

    def resize_content(
        self,
        event,
    ) -> None:
        self.canvas.itemconfigure(
            self.canvas_window,
            width=event.width,
        )

    def mousewheel(
        self,
        event,
    ) -> None:
        self.canvas.yview_scroll(
            int(
                -1
                * (event.delta / 120)
            ),
            "units",
        )

    def get_current_row(self) -> pd.Series:
        return self.conversations.iloc[
            self.current_index
        ]

    def get_current_messages(self) -> pd.DataFrame:
        conversation_id = int(
            self.get_current_row()[
                "conversation_id"
            ]
        )

        return self.messages[
            self.messages[
                "conversation_id"
            ]
            == conversation_id
        ].sort_values(
            [
                "created_at",
                "tweet_id",
            ]
        )


    def load_current_conversation(self) -> None:
        row = self.get_current_row()

        conversation_id = int(
            row["conversation_id"]
        )

        self.conversation_label.config(
            text=(
                f"Conversation ID: "
                f"{conversation_id}"
            )
        )

        candidate_intent = str(
            row.get(
                "candidate_intent_screening",
                "",
            )
        )

        candidate_score = str(
            row.get(
                "candidate_score_screening",
                "",
            )
        )

        review_reason = str(
            row.get(
                "review_pool_reason",
                "",
            )
        )

        self.screening_text.set(
            (
                "Candidate intent: "
                f"{candidate_intent or '(none)'}"
                "   |   Score: "
                f"{candidate_score or '(none)'}"
                "   |   Review reason: "
                f"{review_reason}"
            )
        )

        self.conversation_text.delete(
            "1.0",
            tk.END,
        )

        messages = (
            self.get_current_messages()
        )

        for _, message in messages.iterrows():
            speaker = str(
                message["speaker"]
            )

            tweet_id = int(
                message["tweet_id"]
            )

            timestamp = str(
                message["created_at"]
            )

            text = str(
                message["text"]
            )

            self.conversation_text.insert(
                tk.END,
                (
                    f"[{message['message_number']}] "
                    f"{speaker} | tweet_id={tweet_id}\n"
                ),
            )

            self.conversation_text.insert(
                tk.END,
                f"{timestamp}\n",
            )

            self.conversation_text.insert(
                tk.END,
                f"{text}\n\n",
            )

        status = str(
            row.get(
                "annotation_status",
                "",
            )
        ).strip()

        if status == "annotated":
            self.target_id_var.set(
                str(
                    row.get(
                        "evaluation_incoming_tweet_id",
                        "",
                    )
                )
            )

            self.target_message_text.delete(
                "1.0",
                tk.END,
            )

            self.target_message_text.insert(
                "1.0",
                str(
                    row.get(
                        "evaluation_incoming_message",
                        "",
                    )
                )
            )

            self.intent_var.set(
                str(
                    row.get(
                        "human_intent",
                        "",
                    )
                )
            )

            self.confidence_var.set(
                str(
                    row.get(
                        "annotator_confidence",
                        "",
                    )
                )
            )

            self.multi_var.set(
                str(
                    row.get(
                        "is_multi_intent",
                        "",
                    )
                )
            )

            self.secondary_var.set(
                str(
                    row.get(
                        "secondary_issue_present",
                        "",
                    )
                )
            )

            self.notes_text.delete(
                "1.0",
                tk.END,
            )

            self.notes_text.insert(
                "1.0",
                str(
                    row.get(
                        "annotation_notes",
                        "",
                    )
                )
            )

        else:
            self.clear_human_fields()

        self.refresh_progress()

        self.status_label.config(
            text=(
                "Loaded candidate "
                f"{self.current_index + 1}"
            )
        )

        self.canvas.yview_moveto(0)

    def refresh_progress(self) -> None:
        annotated = int(
            (
                self.conversations[
                    "annotation_status"
                ]
                == "annotated"
            ).sum()
        )

        total = len(
            self.conversations
        )

        self.progress_label.config(
            text=(
                f"Candidate "
                f"{self.current_index + 1} / {total}"
                f"   |   Annotated: "
                f"{annotated} / {total}"
            )
        )
    def clear_human_fields(self) -> None:
        self.target_id_var.set("")

        self.target_message_text.delete(
            "1.0",
            tk.END,
        )

        self.intent_var.set("")
        self.confidence_var.set("")
        self.multi_var.set("")
        self.secondary_var.set("")

        self.notes_text.delete(
            "1.0",
            tk.END,
        )

    def load_exact_customer_message(self) -> None:
        value = (
            self.target_id_var
            .get()
            .strip()
        )

        if not value:
            messagebox.showwarning(
                "Missing tweet ID",
                "Enter the customer tweet ID first.",
            )
            return

        try:
            tweet_id = int(value)
        except ValueError:
            messagebox.showwarning(
                "Invalid tweet ID",
                "Tweet ID must be numeric.",
            )
            return

        messages = (
            self.get_current_messages()
        )

        matching = messages[
            messages["tweet_id"]
            == tweet_id
        ]

        if matching.empty:
            messagebox.showwarning(
                "Tweet not found",
                (
                    "That tweet ID is not part of "
                    "the current conversation."
                ),
            )
            return

        row = matching.iloc[0]

        if not bool(
            row["is_customer_message"]
        ):
            messagebox.showwarning(
                "Wrong speaker",
                (
                    "That tweet is not a customer message. "
                    "Please select an incoming CUSTOMER tweet."
                ),
            )
            return

        exact_text = str(
            row["text"]
        )

        self.target_message_text.delete(
            "1.0",
            tk.END,
        )

        self.target_message_text.insert(
            "1.0",
            exact_text,
        )

        self.status_label.config(
            text=(
                f"Loaded tweet {tweet_id}"
            )
        )

    def validate_annotation(self) -> bool:
        target_id_text = (
            self.target_id_var
            .get()
            .strip()
        )

        intent = (
            self.intent_var
            .get()
            .strip()
        )

        confidence = (
            self.confidence_var
            .get()
            .strip()
        )

        multi = (
            self.multi_var
            .get()
            .strip()
        )

        secondary = (
            self.secondary_var
            .get()
            .strip()
        )

        if not target_id_text:
            messagebox.showwarning(
                "Missing tweet ID",
                (
                    "Enter the tweet ID of the genuine "
                    "incoming customer support message."
                ),
            )
            return False

        try:
            target_id = int(
                target_id_text
            )
        except ValueError:
            messagebox.showwarning(
                "Invalid tweet ID",
                "Tweet ID must be numeric.",
            )
            return False

        current_messages = (
            self.get_current_messages()
        )

        matching = current_messages[
            current_messages[
                "tweet_id"
            ]
            == target_id
        ]

        if matching.empty:
            messagebox.showwarning(
                "Tweet not found",
                (
                    "That tweet ID is not part of "
                    "the current conversation."
                ),
            )
            return False

        target_row = matching.iloc[0]

        if not bool(
            target_row["is_customer_message"]
        ):
            messagebox.showwarning(
                "Wrong speaker",
                (
                    "The evaluation message must be "
                    "an incoming CUSTOMER message."
                ),
            )
            return False

        exact_text = str(
            target_row["text"]
        )

        self.target_message_text.delete(
            "1.0",
            tk.END,
        )

        self.target_message_text.insert(
            "1.0",
            exact_text,
        )

        if intent not in (
            TARGET_INTENTS
            + REVIEW_ONLY
        ):
            messagebox.showwarning(
                "Missing intent",
                (
                    "Select one of the eight frozen "
                    "target intents, other_unclear, "
                    "or non_support_social."
                ),
            )
            return False

        if confidence not in CONFIDENCE_OPTIONS:
            messagebox.showwarning(
                "Missing confidence",
                (
                    "Select high, medium, or low."
                ),
            )
            return False

        if multi not in YES_NO:
            messagebox.showwarning(
                "Missing multi-intent field",
                "Select yes or no.",
            )
            return False

        if secondary not in YES_NO:
            messagebox.showwarning(
                "Missing secondary-issue field",
                "Select yes or no.",
            )
            return False

        return True

    def save_current(
        self,
        silent: bool = False,
    ) -> bool:
        if not self.validate_annotation():
            return False

        index = self.current_index

        target_id = (
            self.target_id_var
            .get()
            .strip()
        )

        exact_message = (
            self.target_message_text
            .get(
                "1.0",
                tk.END,
            )
            .strip()
        )

        self.conversations.at[
            index,
            "evaluation_incoming_tweet_id",
        ] = target_id

        self.conversations.at[
            index,
            "evaluation_incoming_message",
        ] = exact_message

        self.conversations.at[
            index,
            "human_intent",
        ] = (
            self.intent_var
            .get()
            .strip()
        )

        self.conversations.at[
            index,
            "annotator_confidence",
        ] = (
            self.confidence_var
            .get()
            .strip()
        )

        self.conversations.at[
            index,
            "is_multi_intent",
        ] = (
            self.multi_var
            .get()
            .strip()
        )

        self.conversations.at[
            index,
            "secondary_issue_present",
        ] = (
            self.secondary_var
            .get()
            .strip()
        )

        self.conversations.at[
            index,
            "annotation_notes",
        ] = (
            self.notes_text
            .get(
                "1.0",
                tk.END,
            )
            .strip()
        )

        self.conversations.at[
            index,
            "annotation_status",
        ] = "annotated"

        self.conversations.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        self.refresh_progress()

        self.status_label.config(
            text="Annotation saved",
        )

        if not silent:
            messagebox.showinfo(
                "Saved",
                (
                    "Annotation saved to the working file.\n\n"
                    "The final 200-example golden dataset "
                    "has NOT been created."
                ),
            )

        return True

    def save_and_next(self) -> None:
        if not self.save_current(
            silent=True
        ):
            return

        if (
            self.current_index
            < len(
                self.conversations
            ) - 1
        ):
            self.current_index += 1
            self.load_current_conversation()
        else:
            messagebox.showinfo(
                "Finished",
                (
                    "You reached the end of the "
                    "360-candidate review pool.\n\n"
                    "The final 200-example golden "
                    "dataset has NOT been created."
                ),
            )
    def previous_conversation(self) -> None:
        if self.current_index <= 0:
            return

        self.current_index -= 1
        self.load_current_conversation()

    def jump_to(self) -> None:
        value = (
            self.jump_var
            .get()
            .strip()
        )

        if not value:
            return

        try:
            candidate_number = int(
                value
            )
        except ValueError:
            messagebox.showwarning(
                "Invalid number",
                "Enter a candidate number from 1 to 360.",
            )
            return

        index = candidate_number - 1

        if (
            index < 0
            or index >= len(
                self.conversations
            )
        ):
            messagebox.showwarning(
                "Out of range",
                (
                    "Enter a candidate number from 1 to "
                    f"{len(self.conversations)}."
                ),
            )
            return

        self.current_index = index

        self.load_current_conversation()

    def exit_tool(self) -> None:
        self.conversations.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        self.root.destroy()


def main() -> None:
    conversations, messages = (
        load_data()
    )

    print(
        f"Loaded {len(conversations):,} "
        "candidate conversations."
    )

    print(
        "Working annotation file:"
    )

    print(
        OUTPUT_PATH
    )

    root = tk.Tk()

    AnnotationTool(
        root,
        conversations,
        messages,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
