import { createBrowserRouter, Navigate } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { LoginPage } from "./routes/LoginPage";
import { ClientNewRequestPage } from "./routes/client/ClientNewRequestPage";
import { ClientRequestDetailPage } from "./routes/client/ClientRequestDetailPage";
import { BankerRequestsPage } from "./routes/banker/BankerRequestsPage";
import { BankerRequestDetailPage } from "./routes/banker/BankerRequestDetailPage";
import { AuthGuard } from "./routes/guards/AuthGuard";
import { RoleGuard } from "./routes/guards/RoleGuard";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppLayout />, 
    children: [
      { index: true, element: <Navigate to="/login" replace /> },
      { path: "login", element: <LoginPage /> },
      {
        path: "client",
        element: (
          <AuthGuard>
            <RoleGuard allow={["client"]}>
              <AppLayout />
            </RoleGuard>
          </AuthGuard>
        ),
        children: [
          { path: "requests/new", element: <ClientNewRequestPage /> },
          { path: "requests/:id", element: <ClientRequestDetailPage /> },
        ],
      },
      {
        path: "banker",
        element: (
          <AuthGuard>
            <RoleGuard allow={["banker"]}>
              <AppLayout />
            </RoleGuard>
          </AuthGuard>
        ),
        children: [
          { path: "requests", element: <BankerRequestsPage /> },
          { path: "requests/:id", element: <BankerRequestDetailPage /> },
        ],
      },
    ],
  },
]);
