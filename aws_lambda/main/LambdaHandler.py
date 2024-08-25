import boto3
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID
model_id = "anthropic.claude-v2"

# Start a conversation with the user message.
user_message = """

Human:
<text>
 Joe: Hi Hannah!
 Hannah: Hi Joe! Are you coming over? 
 Joe: Yup! Hey I, uh, forgot where you live." 
 Hannah: No problem! It's 0000 Pacos Ln, Los Getos CA 00000.
 Joe: Got it, thanks! 
</text> 

Please remove all personally identifying information from this text and replace it with a single “XXX”. It's very important that nick names, first names, last names, phone numbers, and email addresses get replaced with XXX. 
Please mask out names associated with each message as well.
Please output your sanitized version of the text with PII removed in <response></response> XML tags.

Assistant:"""
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]


def lambda_handler(event, context):
    
    try:
        print("Boto3 version = " + boto3.__version__)
            
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId="anthropic.claude-v2",
            messages=conversation,
            inferenceConfig={"maxTokens":2048,"stopSequences":["\n\nHuman:"],"temperature":0.5,"topP":1},
            additionalModelRequestFields={"top_k":250}
        )

        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        print(response_text)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)
