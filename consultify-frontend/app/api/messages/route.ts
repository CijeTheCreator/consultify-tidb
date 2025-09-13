import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function POST(request: NextRequest) {
  try {
    const {
      senderId,
      consultationId,
      originalContent,
      originalLanguage,
      translatedContent,
      translatedLanguage,
      llm_content,
      llm_language,
      state
    } = await request.json()

    if (!senderId || !consultationId) {
      return NextResponse.json(
        { error: 'SenderId and consultationId are required' },
        { status: 400 }
      )
    }

    const consultation = await prisma.consultation.findUnique({
      where: { id: consultationId },
    })

    if (!consultation) {
      console.log("Problem: Consultation not found")
      return NextResponse.json(
        { error: 'Consultation not found' },
        { status: 404 }
      )
    }

    const sender = await prisma.user.findUnique({
      where: { id: senderId },
    })

    if (!sender) {
      return NextResponse.json(
        { error: 'Sender not found' },
        { status: 404 }
      )
    }

    const message = await prisma.message.create({
      data: {
        senderId,
        consultationId,
        originalContent: originalContent || null,
        originalLanguage: originalLanguage || null,
        translatedContent: translatedContent || null,
        translatedLanguage: translatedLanguage || null,
        llm_content: llm_content || "en",
        llm_language: llm_language || "en",
        state: state || null,
      },
      include: {
        sender: true,
        consultation: true,
      },
    })

    return NextResponse.json(message)
  } catch (error) {
    console.error('Error creating message:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
