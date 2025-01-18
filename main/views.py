from django.shortcuts import render
from django.http import JsonResponse
from openai import OpenAI
from django.conf import settings


client = OpenAI(api_key=settings.OPENAI_API_KEY)

def index(request):
    return render(request, 'index.html')

def travel(request):
    system = '''
    # Role
    A time machine that shows what could have happened if a key event had changed.
    # Task
    Create a picture of the world with a changed history based on the butterfly effect.
    # Input data
    You get the year and the event that the user changed this year.
    # Output data
    - You describe what happened next, how the world changed in the future, describe the changes in key events up to our time.
    - Your text should be in the language of the user's query.
    - Do not use Markdown at all
    - At the beginning of each paragraph, add 6 pieces of &nbsp;
    - At the end of each paragraph, add 2 times <br>
    - Use an artistic style
    # Comments and suggestions
    - The story should be exciting. Make it look like reality, but at the same time, the user's attention should be held with each line.
    - The story can be both positive and negative.
    # Limitation
    - The text should contain a minimum of 500 tokens to a maximum of 1200.
    '''

    year = request.GET.get('year')
    event = request.GET.get('event')
    user = f'Год: {year}, Событие: {event}'
    text = ai_request(system, user)

    middle = len(text) // 2
    text_1 = text[:middle]
    text_2 = text[middle:]

    photo_1 = photo_generate(text_1)
    photo_2 = photo_generate(text_2)

    return JsonResponse({
        'text': text,
        'year': year,
        'event': event,
        'photo_1': photo_1,
        'photo_2': photo_2
    })

def ai_request(system, user):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {'role':'system', 'content': system},
        {'role':'user', 'content': user}
    ],
    response_format={
        "type": "text"
    },
    temperature=1,
    max_tokens=10000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].message.content.strip()

def photo_generate(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    return response.data[0].url