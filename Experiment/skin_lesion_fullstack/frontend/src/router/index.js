import { createRouter, createWebHistory } from "vue-router";
import MainLayout from "../layouts/MainLayout.vue";

const routes = [
  {
    path: "/",
    component: MainLayout,
    redirect: "/single",
    children: [
      { path: "single", name: "single", component: () => import("../views/SingleDetect.vue") },
      { path: "batch", name: "batch", component: () => import("../views/BatchDetect.vue") },
      { path: "history", name: "history", component: () => import("../views/History.vue") },
      { path: "settings", name: "settings", component: () => import("../views/Settings.vue") },
    ],
  },
];

export default createRouter({
  history: createWebHistory(),
  routes,
});
