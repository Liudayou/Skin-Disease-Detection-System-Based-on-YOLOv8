/** 与论文描述一致的类别配色（英文类名键） */
export const CLASS_COLORS = {
  nv: "#F56C6C",
  mel: "#E6A23C",
  bcc: "#F2D024",
  bkl: "#67C23A",
  akiec: "#13C2C2",
  vasc: "#409EFF",
  df: "#9B59B6",
};

export function colorForClass(nameEn) {
  return CLASS_COLORS[nameEn] || "#909399";
}
