import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const messageId = params.id

    const message = await prisma.message.findUnique({
      where: { id: messageId },
      include: {
        sender: true,
        consultation: true,
      },
    })

    if (!message) {
      return NextResponse.json(
        { error: 'Message not found' },
        { status: 404 }
      )
    }

    return NextResponse.json(message)
  } catch (error) {
    console.error('Error fetching message:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const messageId = params.id
    const { 
      originalContent, 
      originalLanguage, 
      translatedContent, 
      translatedLanguage, 
      llm_content, 
      llm_language,
      state 
    } = await request.json()

    const message = await prisma.message.findUnique({
      where: { id: messageId },
    })

    if (!message) {
      return NextResponse.json(
        { error: 'Message not found' },
        { status: 404 }
      )
    }

    const updateData: any = {}
    
    if (originalContent !== undefined) updateData.originalContent = originalContent
    if (originalLanguage !== undefined) updateData.originalLanguage = originalLanguage
    if (translatedContent !== undefined) updateData.translatedContent = translatedContent
    if (translatedLanguage !== undefined) updateData.translatedLanguage = translatedLanguage
    if (llm_content !== undefined) updateData.llm_content = llm_content
    if (llm_language !== undefined) updateData.llm_language = llm_language
    if (state !== undefined) updateData.state = state

    const updatedMessage = await prisma.message.update({
      where: { id: messageId },
      data: updateData,
      include: {
        sender: true,
        consultation: true,
      },
    })

    return NextResponse.json(updatedMessage)
  } catch (error) {
    console.error('Error updating message:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}