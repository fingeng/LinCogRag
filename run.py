import os
import sys

# 保持兼容：允许直接 python run.py 在项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import parse_args
from src.pipeline import run


def main():
    args = parse_args()
    _, summary, results_path, summary_path = run(args)

    print("\n======================================================================")
    print("📈 Overall Results")
    print("======================================================================")
    print(f"Total questions:         {summary['total_questions']}")
    print(f"LLM Accuracy:            {summary['overall_llm_accuracy']:.2f}% ({summary['total_correct']}/{summary['total_questions']})")
    print(f"Contain Accuracy:        {summary['overall_contain_accuracy']:.2f}% ({summary['total_contain_correct']}/{summary['total_questions']})")
    print(f"Questions w/o entities:  {summary['questions_wo_entities']} ({(summary['questions_wo_entities']/summary['total_questions']*100 if summary['total_questions'] else 0):.1f}%)")
    print(f"Invalid answers:         {summary['total_invalid']}")
    print(f"Valid answer rate:       {summary['valid_answer_rate']:.2f}%")
    print("======================================================================\n")
    print(f"💾 Detailed results saved to: {results_path}")
    print(f"💾 Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()