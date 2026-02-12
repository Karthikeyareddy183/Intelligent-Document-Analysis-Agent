"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listConversations,
  getConversation,
  createConversation,
  deleteConversation,
} from "@/lib/api";

export function useConversations(documentId?: string) {
  return useQuery({
    queryKey: ["conversations", documentId],
    queryFn: () => listConversations(documentId),
    enabled: !!documentId,
  });
}

export function useConversation(conversationId: string | null) {
  return useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => getConversation(conversationId!),
    enabled: !!conversationId,
  });
}

export function useCreateConversation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      documentId,
      title,
    }: {
      documentId: string;
      title?: string;
    }) => createConversation(documentId, title),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["conversations", variables.documentId],
      });
    },
  });
}

export function useDeleteConversation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteConversation,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
}
